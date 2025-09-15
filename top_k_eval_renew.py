import os
import itertools
import random
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import argparse
import matplotlib.pyplot as plt
import logging
import torchvision.transforms.v2 as v2
import numpy as np
from multiprocessing.pool import Pool
from datetime import datetime
from torch.utils.data import Dataset , DataLoader
import gc

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

print(f"script dir ---> {script_dir}")

class Dataset_load(Dataset):
    def __init__(self, identity_map):
        super().__init__()
        self.all_images = sorted(list(set(itertools.chain.from_iterable(identity_map.values()))))

        self.transform = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop(size=(112, 112)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        image_path = self.all_images[index]
        image = cv2.imread(image_path) # BGR순서로 읽음..
        #image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB) # adaface 스킵
        image_tensor = self.transform(image)
        return image_tensor, image_path
    
def find_max_batch_size(model, input_shape, device):
    if device != 'cuda':
        logging.info("CUDA 발견 실패 MAX Batch SIZE 탐색 실패")
        return None
    
    model.to(device)
    model.eval()

    batch_size = 512
    max_batch_size = 0

    while True:
        try:
            dummy_input = torch.randn(batch_size, *input_shape).to(device)

            with torch.no_grad():
                _ = model(dummy_input)
            max_batch_size = batch_size
            print(f"✅ 배치 사이즈 {batch_size} 성공")
            batch_size *= 2 

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"❌ 배치 사이즈 {batch_size}에서 메모리 부족 발생")
                torch.cuda.empty_cache()
                break
            else:
                raise e
            
        if batch_size > 2048 * 5:
                print("안전 제한 도달. 탐색을 중단합니다.")
                break
        
    return max_batch_size


def init_identification_worker(worker_embeddings, gallery_embs_np, gallery_ids):

    global g_embeddings, g_gallery_embeddings_np, g_gallery_identities_ordered
    g_embeddings = worker_embeddings
    g_gallery_embeddings_np = gallery_embs_np
    g_gallery_identities_ordered = gallery_ids

def _evaluate_probe_worker(probe_data):
    global g_embeddings, g_gallery_embeddings_np, g_gallery_identities_ordered
    probe_img_path, true_identity = probe_data

    probe_emb = g_embeddings.get(probe_img_path)
    if probe_emb is None:
        return -1  

    # Convert probe to float32 numpy array for calculation
    probe_emb_f32 = probe_emb.numpy().astype(np.float32)
    
    norm_val = np.linalg.norm(probe_emb_f32)
    if norm_val == 0:
        return -1
    probe_emb_norm = probe_emb_f32 / norm_val

    if not np.all(np.isfinite(probe_emb_norm)):
        return -1

    # g_gallery_embeddings_np is already float32 from calculate_identification_metrics
    similarities = np.dot(g_gallery_embeddings_np, probe_emb_norm)
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_identities = np.array(g_gallery_identities_ordered)[ranked_indices]
    match_indices = np.where(ranked_identities == true_identity)[0]

    if len(match_indices) > 0:
        return match_indices[0] + 1  # Return 1-based rank
    else:
        return -1 # Not found

def init_worker(worker_embeddings):
    global embeddings
    embeddings = worker_embeddings

transforms_v2 = v2.Compose([
    v2.ToImage(), # cv2 HWC ---> Tensor C H W
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=(112, 112)),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@torch.inference_mode()
def get_all_embeddings(identity_map, backbone, batch_size):
    
    logging.info(f"임베딩 추출 시작 ( 배치사이즈: {batch_size})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"현재 사용 장치: {device}")
    logging.info(f"장치 타입: {device.type}")

    if device.type == 'cpu':
        input("\nCPU 추론을 진행합니다. 계속하시려면 아무 키나 입력하세요...")

    backbone = backbone.to(device)
    backbone.eval()
    
    embeddings = {} 
    dataset = Dataset_load(identity_map)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True, 
    )


    for batch_tensor, batch_paths in tqdm(dataloader, desc='임베딩 추출'):
        if batch_tensor is None:
            continue

        batch_tensor = batch_tensor.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            output = backbone(batch_tensor)
            vectors = output[1] if isinstance(output, tuple) else output
            
        if vectors is not None and vectors.numel() > 0:
            vectors_cpu = vectors.cpu()
            for path, vector in zip(batch_paths, vectors_cpu):
                embeddings[path] = vector.to(torch.float16).flatten()


    try:
        if args.save_cache:
            file_name = f'{args.model}.npz'
            embeddings_np = {k: v.cpu().numpy() for k, v in embeddings.items()}
            np.savez_compressed(file_name, **embeddings_np)
            logging.info(f"임베딩 캐시 저장완료 파일이름 : {file_name}")
        
    except Exception as e:
        logging.info(f'{e}')
        logging.info("@@임베딩 캐시 저장 실패 코드검수.....!!@@")

    return embeddings

def _calculate_similarity_for_pair(pair):
    global embeddings
    img1_path, img2_path = pair
    
    emb1 = embeddings.get(img1_path)
    emb2 = embeddings.get(img2_path)
    
    if emb1 is not None and emb2 is not None:
        emb1 = emb1.to(torch.float32)
        emb2 = emb2.to(torch.float32)

        norm1 = torch.norm(emb1)
        norm2 = torch.norm(emb2)

        if norm1 > 0 and norm2 > 0:
            cosine_similarity = torch.dot(emb1, emb2) / (norm1 * norm2)
            return cosine_similarity.cpu().numpy().astype(np.float16)
            
    return None

def calculate_identification_metrics(identity_map, embeddings):
    logging.info("Calculating identification metrics (Rank-k, CMC)...")

    gallery_images = {} 
    probe_images_with_labels = []


    REPRESENTATIVE_IMAGE_INDEX = 2098
    CLASS_LEN = len(next(iter(identity_map.values())))

    if CLASS_LEN < REPRESENTATIVE_IMAGE_INDEX:
        logging.info(f"{CLASS_LEN} has fewer than {REPRESENTATIVE_IMAGE_INDEX} images; using the first image for the gallery.")

    for identity, img_paths in identity_map.items():
        if not img_paths:
            continue

        try:
            gallery_images[identity] = img_paths[REPRESENTATIVE_IMAGE_INDEX]
            idx = REPRESENTATIVE_IMAGE_INDEX 

        except IndexError:
            gallery_images[identity] = img_paths[0]
            idx = 0

        for i in range(len(img_paths)):
            if i == idx:
                continue
            probe_images_with_labels.append((img_paths[i], identity))
    
    if not probe_images_with_labels:
        logging.warning("No probe images available for identification evaluation. Skipping identification metrics.")
        return None, None, None, None, None

    logging.info(f"Total Class : {len(gallery_images)}")
    logging.info(f"Total probe images: {len(probe_images_with_labels)}")

    gallery_embeddings = [] 
    gallery_identities_ordered = [] 
    for identity in sorted(gallery_images.keys()): 
        img_path = gallery_images[identity]
        emb = embeddings.get(img_path)
        if emb is not None:
            norm = torch.norm(emb)
            if norm > 0:
                norm_emb = torch.div(emb , norm)
                if torch.all(torch.isfinite(norm_emb)):
                    gallery_embeddings.append(norm_emb)
                    gallery_identities_ordered.append(identity)
                else:
                    logging.warning(f"Non-finite gallery embedding for identity {identity}: {img_path}")
            else:
                logging.warning(f"Zero-norm gallery embedding found for identity {identity}: {img_path}")
        else:
            logging.warning(f"Gallery image embedding missing for identity {identity}: {img_path}")
    
    if not gallery_embeddings:
        logging.error("No valid gallery embeddings found. Cannot perform identification evaluation.")
        return None, None, None, None, None

    gallery_embeddings_np = torch.stack(gallery_embeddings).cpu().numpy().astype(np.float32)

    max_rank = len(gallery_identities_ordered) 
    if max_rank == 0: 
        logging.error("Gallery is empty. Cannot calculate identification metrics.")
        return None, None, None, None, None

    from multiprocessing import Pool, cpu_count

    init_args = (embeddings, gallery_embeddings_np, gallery_identities_ordered)
    all_ranks = []
    with Pool(initializer=init_identification_worker, initargs=init_args, processes=cpu_count()) as pool:
        results_iterator = pool.imap(_evaluate_probe_worker, probe_images_with_labels, chunksize=1000)
        all_ranks = list(tqdm(results_iterator, total=len(probe_images_with_labels), desc="Evaluating identification (multi-process)"))

    valid_ranks = [r for r in all_ranks if r > 0]
    total_probes = len(valid_ranks)

    if total_probes == 0:
        logging.warning("No valid probes were processed.")
        return None, None, None, None, None

    valid_ranks_np = np.array(valid_ranks)
    rank_counts = np.bincount(valid_ranks_np, minlength=max_rank + 1)
    
    rank_1_accuracy = rank_counts[1] / total_probes
    rank_5_accuracy = np.sum(rank_counts[1:6]) / total_probes
    
    cmc_hits = np.cumsum(rank_counts[1:max_rank + 1])
    cmc_curve = cmc_hits / total_probes

    logging.info(f"Rank-1 Accuracy: {rank_1_accuracy:.4f}")
    logging.info(f"Rank-5 Accuracy: {rank_5_accuracy:.4f}")
    logging.info(f"CMC Curve calculated up to rank {max_rank}")

    return rank_1_accuracy, rank_5_accuracy, cmc_curve, max_rank, total_probes

def generate_positive_pairs(identity_map):
    """동일 인물 쌍을 생성하는 제너레이터"""
    for imgs in identity_map.values():
        for pair in itertools.combinations(imgs, 2):
            yield pair

def generate_negative_pairs(identity_map, num_pairs):
    """다른 인물 쌍을 생성하는 제너레이터 (중복 허용)"""
    identities = list(identity_map.keys())
    if len(identities) < 2:
        return

    for _ in range(num_pairs):
        id1, id2 = random.sample(identities, 2)
        img1 = random.choice(identity_map[id1])
        img2 = random.choice(identity_map[id2])
        yield (img1, img2)

def main(args):
    LOG_FILE = os.path.join(script_dir , f'{args.model}_result.log')
    with open(f"{LOG_FILE}" , 'a') as log_file:
        log_file.write(f"\n시작시간 : {datetime.now().strftime('%Y.%m.%d - %H:%M:%S')}\n")

    torch.backends.cudnn.benchmark = True
    np.random.seed(42)
    random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.model}_LOG.log" , mode='w'),
            logging.StreamHandler()
        ]
    )
    
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {args.data_path}")

    from net import Backbone
    if args.model =='adaface_ir101_webface12m':
        Weight_path = 'adaface_ir101_webface12m.ckpt'
        ckpt = torch.load(Weight_path, map_location='cpu')
        backbone = Backbone([112,112], 100, 'ir')

    elif args.model =='adaface_ir101_webface4m':
        Weight_path = 'adaface_ir101_webface4m.ckpt'
        ckpt = torch.load(Weight_path, map_location='cpu')
        backbone = Backbone([112,112], 100, 'ir')

    elif args.model =='adaface_ir101_ms1mv3':
        Weight_path = 'adaface_ir101_ms1mv3.ckpt'
        ckpt = torch.load(Weight_path, map_location='cpu')
        backbone = Backbone([112,112], 100, 'ir')

    elif args.model == 'adaface_ir50_casia':
        Weight = 'adaface_ir50_casia.ckpt'
        ckpt = torch.load(Weight, map_location='cpu')
        backbone = Backbone([112,112],50,'ir')

    else:
        logging.info(f"Select Model {args.model}")
        exit(0)


    load_result = backbone.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key})
    
    logging.info("누락된 가중치 : {}".format(load_result.missing_keys))
    logging.info("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

    if not load_result.missing_keys and not load_result.unexpected_keys:
        logging.info("모델 가중치가 성공적으로 로드되었습니다.")

    flag = str(input("진행시 아무키... (1)종료..."))
    if flag == '1':
        logging.info("종료../")
        exit(0)

    MAX_BATCH_SIZE = find_max_batch_size(backbone , (3,112,112) , device = 'cuda' if torch.cuda.is_available() else 'cpu')
    gc.collect()
    torch.cuda.empty_cache()
    if MAX_BATCH_SIZE is not None:
        args.batch_size = MAX_BATCH_SIZE // 2
        logging.info(f"배치사이즈 변경(최대치) : {MAX_BATCH_SIZE // 2}")

    backbone = torch.compile(backbone)

    ALL_PERSON_FOLDERS = sorted(os.listdir(args.data_path))
    NUM_FOLDER_TO_PROCESS = len(ALL_PERSON_FOLDERS) // args.split
    folders_to_process = ALL_PERSON_FOLDERS[:NUM_FOLDER_TO_PROCESS]

    logging.info(f"사람 클래스 수 : {len(folders_to_process)}")

    identity_map = {} # 사람폴더라벨 : 해당 폴더 사람 이미지 경로
    for person_folder in folders_to_process:
        person_path = os.path.join(args.data_path, person_folder) # 각 사람폴더의 경로
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # 각사람폴더에 들어있는 모든 jpg
            if len(images) > 1:
                identity_map[person_folder] = images
    
    if not identity_map:
        raise ValueError("데이터셋에서 2개 이상의 이미지를 가진 인물을 찾지 못했습니다.")
    
    TOTAL_IMAGE_LEN = sum(len(v) for v in identity_map.values() if v is not None)
    logging.info(f"총 {len(identity_map)}명의 인물, {sum(len(v) for v in identity_map.values())}개의 이미지를 찾았습니다.")

    logging.info("\n평가에 사용할 동일 인물/다른 인물 쌍 생성을 준비합니다 (제너레이터 사용)...")

    
    num_positive_pairs = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in identity_map.values())
    num_negative_pairs = num_positive_pairs  

    logging.info(f"- 동일 인물 쌍 (Generator 생성..): {num_positive_pairs}개, 다른 인물 쌍 (Generator 생성..): {num_negative_pairs}개")

    positive_pairs_generator = generate_positive_pairs(identity_map)
    negative_pairs_generator = generate_negative_pairs(identity_map, num_negative_pairs)


    if args.load_cache is not None :
        cache_path = args.load_cache
        with np.load(cache_path) as loaded_npz:
            embeddings = {key: torch.from_numpy(loaded_npz[key]) for key in tqdm(loaded_npz.files , desc='임베딩 캐시 로딩..')}

    else:
        embeddings = get_all_embeddings(
            identity_map, backbone ,args.batch_size
        )


    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    import time
    start_time = time.time()
    with Pool(initializer=init_worker, initargs=(embeddings,)) as pool:
        with open(os.path.join(script_dir, 'similarity_for_pair.npy') , 'wb') as f:
            pos_results_gen = pool.imap_unordered(_calculate_similarity_for_pair, positive_pairs_generator, chunksize=1000)
            for result in tqdm(pos_results_gen, total=num_positive_pairs, desc="동일 인물 쌍 계산 및 저장"):
                if result is not None:
                    result.tofile(f)

        with open(os.path.join(script_dir , 'negative_for_pair.npy'), 'wb') as f:
            neg_results_gen = pool.imap_unordered(_calculate_similarity_for_pair, negative_pairs_generator, chunksize=1000)
            for result in tqdm(neg_results_gen, total=num_negative_pairs, desc="다른 인물 쌍 계산 및 저장"):
                if result is not None:
                    result.tofile(f)

    end_time = time.time() - start_time
    logging.info(f"유사도 계산 및 파일 작성 완료. 소요시간: {end_time:.5f}초")

    rank_1_accuracy, rank_5_accuracy, cmc_curve, max_rank, total_probes = calculate_identification_metrics(identity_map, embeddings)

    if rank_1_accuracy is not None:
            print(f"\n--- 얼굴 식별 성능 ---")
            print(f"Rank-1 Accuracy: {rank_1_accuracy:.4f}")
            print(f"Rank-5 Accuracy: {rank_5_accuracy:.4f}")
            print(f"총 프로브 이미지 수: {total_probes}")

            with open(LOG_FILE, 'a') as log_file:
                log_file.write(f"\n얼굴 식별 성능:\n")
                log_file.write(f"Rank-1 Accuracy: {rank_1_accuracy:.4f}\n")
                log_file.write(f"Rank-5 Accuracy: {rank_5_accuracy:.4f}\n")
                log_file.write(f"총 프로브 이미지 수: {total_probes}\n")
                if cmc_curve is not None:
                    log_file.write(f"CMC Curve (first 5 ranks): {cmc_curve[:5].tolist()}\n")
                log_file.write("\n")

            if cmc_curve is not None and max_rank > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(1, max_rank + 1), cmc_curve, marker='o', linestyle='-', markersize=4)
                plt.xlim([1, min(max_rank, 20)]) # Show up to rank 20 or max_rank
                plt.ylim([0.0, 1.05])
                plt.xlabel('Rank (k)')
                plt.ylabel('Accuracy')
                plt.title(f'CMC Curve for {args.model}')
                plt.grid(True)
                cmc_plot_filename = f"_{args.model}_cmc_curve.png"
                plt.savefig(cmc_plot_filename)
                print(f"CMC 커브 그래프가 '{cmc_plot_filename}' 파일로 저장되었습니다.")
    else:
        print("\n--- 얼굴 식별 성능 ---")
        print("얼굴 식별 성능을 계산할 수 없습니다 (유효한 프로브 이미지 부족).")

    num_total_embeddings = len(embeddings)
    num_valid_embeddings = sum(1 for v in embeddings.values() if v is not None)
    num_none_embeddings = sum(1 for v in embeddings.values() if v is None)

    del embeddings
    gc.collect()

    start_time = time.time()
    logging.info("파일에서 유사도 데이터를 로딩합니다 ...")
    try:
        pos_similarities = np.fromfile('similarity_for_pair.npy', dtype=np.float16)
        logging.info(f"동일 인물 유사도 로딩 완료: {len(pos_similarities)}개")

        neg_similarities = np.fromfile('negative_for_pair.npy', dtype=np.float16)
        logging.info(f"다른 인물 유사도 로딩 완료: {len(neg_similarities)}개")

    except FileNotFoundError as e:
        logging.error(f"오류: {e}.")
        exit(1)

    end_time = time.time() - start_time
    logging.info(f"파일 로드 소요시간: {end_time:.5f}초")

    pos_labels = np.ones(len(pos_similarities))
    neg_labels = np.zeros(len(neg_similarities))

    pos_finite_mask = np.isfinite(pos_similarities)
    neg_finite_mask = np.isfinite(neg_similarities)

    pos_similarities = pos_similarities[pos_finite_mask]
    neg_similarities = neg_similarities[neg_finite_mask]
    pos_labels = pos_labels[pos_finite_mask]
    neg_labels = neg_labels[neg_finite_mask]


    print(f"\n--- 유사도 분포 분석 ---")
    if len(pos_similarities) > 0 and len(neg_similarities) > 0:
        def safe_std(arr):
            try:
                arr_f64 = arr.astype(np.float64)
                std_val = np.std(arr_f64, dtype=np.float64)
                return std_val if np.isfinite(std_val) else "overflow"
            except:
                return "계산 불가"
        
        pos_std = safe_std(pos_similarities)
        neg_std = safe_std(neg_similarities)
        
        print(f"🔵 동일 인물 쌍 유사도 (총 {len(pos_similarities):,}개):")
        print(f"   - 최소값: {np.min(pos_similarities):.4f}")
        print(f"   - 최대값: {np.max(pos_similarities):.4f}")
        print(f"   - 평균값: {np.mean(pos_similarities.astype(np.float64)):.4f}")
        print(f"   - 표준편차: {pos_std:.4f}" if isinstance(pos_std, (int, float)) else f"   - 표준편차: {pos_std}")
        
        print(f"🔴 다른 인물 쌍 유사도 (총 {len(neg_similarities):,}개):")
        print(f"   - 최소값: {np.min(neg_similarities):.4f}")
        print(f"   - 최대값: {np.max(neg_similarities):.4f}")
        print(f"   - 평균값: {np.mean(neg_similarities.astype(np.float64)):.4f}")
        print(f"   - 표준편차: {neg_std:.4f}" if isinstance(neg_std, (int, float)) else f"   - 표준편차: {neg_std}")



        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\n--- 유사도 분포 분석 ---\n")  
            log_file.write(f"🔍 디버깅 정보:\n")          
            log_file.write(f"   - 전체 임베딩 수: {num_total_embeddings}\n")
            log_file.write(f"   - 유효한 임베딩 수: {num_valid_embeddings}\n")
            log_file.write(f"   - None 임베딩 수: {num_none_embeddings}\n")
            log_file.write(f"\n") 

            log_file.write(f"🔵 동일 인물 쌍 유사도 (총 {len(pos_similarities):,}개):\n")
            log_file.write(f"   - 최소값: {np.min(pos_similarities):.4f}\n")     
            log_file.write(f"   - 최대값: {np.max(pos_similarities):.4f}\n")        
            log_file.write(f"   - 평균값: {np.mean(pos_similarities.astype(np.float64)):.4f}\n") 
            
            if isinstance(pos_std, (int, float)):
                log_file.write(f"   - 표준편차: {pos_std:.4f}\n")
            else:
                log_file.write(f"   - 표준편차: {pos_std}\n")
            
            log_file.write(f"\n")
            log_file.write(f"🔴 다른 인물 쌍 유사도 (총 {len(neg_similarities):,}개):\n") 
            log_file.write(f"   - 최소값: {np.min(neg_similarities):.4f}\n")        
            log_file.write(f"   - 최대값: {np.max(neg_similarities):.4f}\n")         
            log_file.write(f"   - 평균값: {np.mean(neg_similarities.astype(np.float64)):.4f}\n") 

            if isinstance(neg_std, (int, float)):
                log_file.write(f"   - 표준편차: {neg_std:.4f}\n")
            else:
                log_file.write(f"   - 표준편차: {neg_std}\n")
            
            log_file.write('\n')

    else:
        logging.info("유사도 데이터가 충분하지 않아 분포 분석을 수행할 수 없습니다.")
        logging.info(f"len pos : {len(pos_similarities)}, len neg: {len(neg_similarities)}")
        exit(0)

    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"   - 양성 쌍 유사도 수 (필터링 후): {len(pos_similarities)}\n")
        log_file.write(f"   - 음성 쌍 유사도 수 (필터링 후): {len(neg_similarities)}\n")
    


    scores = np.concatenate([pos_similarities, neg_similarities])
    labels = np.concatenate([pos_labels, neg_labels])

    del pos_similarities , neg_similarities , pos_labels , neg_labels
    gc.collect()

    logging.info("\n--- 최종 평가 결과 ---")
    if labels.size > 0:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        frr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - frr))
        eer = fpr[eer_index]
        eer_threshold = thresholds[eer_index]

        tar_at_far_results = {far: np.interp(far, fpr, tpr) for far in args.target_fars}
        
        predictions = (scores >= eer_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {"accuracy": accuracy, "recall": recall, "f1_score": f1_score, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        print(f"전체 평가 쌍: {len(labels)} 개")
        print(f"[주요 성능] ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (유사도 임계값: {eer_threshold:.4f})")
        print(f"[상세 지표] Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        for far, tar in tar_at_far_results.items():
            print(f"  - TAR @ FAR {far*100:g}%: {tar:.4f}")
        
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\n평가 결과:\n")
            log_file.write(f"전체 클래스수  : {NUM_FOLDER_TO_PROCESS}  전체 사람 이미지수 : {TOTAL_IMAGE_LEN}\n")
            log_file.write(f"ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (Threshold: {eer_threshold:.4f})\n")
            log_file.write(f"Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}\n")
            for far, tar in tar_at_far_results.items():
                log_file.write(f"TAR @ FAR {far*100:g}%: {tar:.4f}\n")
            log_file.write("\n")  
            log_file.write(f"평가를 완료하였습니다 (종료시간) ---> {datetime.now().strftime('%Y.%m.%d - %H:%M:%S')}")
            log_file.write("\n")

        excel_path = os.path.join(script_dir, args.excel_path)
        total_dataset_img_len = sum(len(v) for v in identity_map.values())
        total_class = len(identity_map)

        try:
            plot_roc_curve(fpr, tpr, roc_auc, args.model, excel_path)
        except Exception as e:
            logging.info(f"ROC Curve 이미지 저장 실패 {e}")

        try:
            save_results_to_excel(excel_path, args.model, roc_auc, eer, tar_at_far_results, \
                              args.target_fars, metrics, total_dataset_img_len, total_class, args.data_path, args.model, \
                              rank_1_accuracy, rank_5_accuracy)
        except Exception as e:
            logging.info(f"EXCEL SAVE 저장 실패 {e}")

        logging.info(f"평가를 완료하였습니다 (종료시간) ---> {datetime.now().strftime('%Y.%m.%d - %H:%M:%S')}")

    else:
        msg = "평가를 위한 유효한 점수를 수집하지 못했습니다."
        print(msg)
        logging.error(msg)

def save_results_to_excel(excel_path, model_name, roc_auc, eer, tar_at_far_results, target_fars, metrics, total_dataset_img_len, total_class,
                           data_path, model_attr_value, rank_1_accuracy=None, rank_5_accuracy=None):


    new_data = {
        "model_name": [model_name],
        "roc_auc": [f"{roc_auc:.4f}"], "eer": [f"{eer:.4f}"],
        "accuracy": [f"{metrics['accuracy']:.4f}"], "recall": [f"{metrics['recall']:.4f}"],
        "f1_score": [f"{metrics['f1_score']:.4f}"], "tp": [metrics['tp']],
        "tn": [metrics['tn']], "fp": [metrics['fp']], "fn": [metrics['fn']]
    }

    for far in target_fars:
        new_data[f"tar_at_far_{far*100:g}%"] = [f"{tar_at_far_results.get(far, 0):.4f}"]

    if rank_1_accuracy is not None:
        new_data["rank_1_accuracy"] = [f"{rank_1_accuracy:.4f}"]
    if rank_5_accuracy is not None:
        new_data["rank_5_accuracy"] = [f"{rank_5_accuracy:.4f}"]

    new_data.update({
        'total_dataset_img_len': [total_dataset_img_len],
        'total_class': [total_class],
        'data_path': [data_path],
        'model_attr': [model_attr_value]
    })
    
    new_df = pd.DataFrame(new_data)
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_excel(excel_path, index=False)
    print(f"\n평가 결과가 '{excel_path}' 파일에 저장되었습니다.")

def plot_roc_curve(fpr, tpr, roc_auc, model_name, excel_path):
    """ROC 커브를 그리고 파일로 저장합니다."""

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC Curve {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.splitext(excel_path)[0] + f"_{model_name}_roc_curve.png"
    plt.savefig(plot_filename)
    print(f"ROC 커브 그래프가 '{plot_filename}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEvaluation Script")
    parser.add_argument('--model',type=str , default='adaface_ir101_webface12m', choices=['adaface_ir101_webface12m' , 'adaface_ir101_webface4m','adaface_ir101_ms1mv3','adaface_ir50_casia'],)
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/KOR_DATA/일반/kor_data_sorting", help="평가할 데이터셋의 루트 폴더")
    parser.add_argument("--excel_path", type=str, default="evaluation_results.xlsx", help="결과를 저장할 Excel 파일 이름")
    parser.add_argument("--target_fars", nargs='+', type=float, default=[0.01, 0.001, 0.0001], help="TAR을 계산할 FAR 목표값들")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="사용할 장치 (예: cpu, cuda, cuda:0)")
    parser.add_argument("--batch_size", type=int, default=512, help="임베딩 추출 시 배치 크기")
    parser.add_argument('--load_cache' , type=str , default = None ,help="임베딩 캐시경로")
    parser.add_argument('--save_cache' , action='store_true')
    parser.add_argument('--split', type= int ,default=1 , help='전체클래스수 / N ')
    args = parser.parse_args()

    #args.data_path = '/home/ubuntu/KOR_DATA/kor_data_full_Middle_Resolution_aligend'

    for key , values in args.__dict__.items():
        print(f"key {key}  :  {values}")

    main(args)
