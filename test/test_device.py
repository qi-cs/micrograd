from micrograd.engine import Value
import micrograd_cpu
import micrograd_cuda

# Test CPU operations
a = Value(2.0, device='cpu')
b = Value(3.0, device='cpu')
c = a + b
print(f"CPU Addition: {c.data}")

# Test CUDA operations
a = Value(2.0, device='cuda')
b = Value(3.0, device='cuda')
c = a + b
print(f"CUDA Addition: {c.data}")
