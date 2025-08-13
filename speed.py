import torch
import time
from tqdm import tqdm
def benchmark_model(model, input_shape, device='cuda', warmup_runs=10, test_runs=500):
    model.eval()
    model = model.to(device)
    
    # 더미 입력 생성
    dummy_input = torch.randn(input_shape).to(device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Latency 측정
        latencies = []
        for _ in tqdm(range(test_runs)):
            if device == 'cuda':
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                
                starter.record()
                _ = model(dummy_input)
                ender.record()
                
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
            else:
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        # 통계 계산
        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000 / avg_latency  # FPS
        
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Throughput: {throughput:.2f} FPS")
        
    return avg_latency, throughput

from net import Backbone

backbone = Backbone([112,112], 50, 'ir')
latency, fps = benchmark_model(backbone, (1, 3, 112, 112))
