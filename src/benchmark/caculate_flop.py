"""
Benchmark module for calculating FLOPs and parameters of neural network models.
Cung cấp class để đo lường hiệu suất tính toán của mô hình.
"""

import torch
from pathlib import Path
from thop import profile
from typing import Dict, Tuple, Optional


# ============================================================================
# CONSTANTS
# ============================================================================

UNIT_CONVERSION = {
    "params": 1_000_000,      # Convert to Millions
    "flops": 1_000_000_000,   # Convert to Giga
}

DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)  # (Batch, Channels, Height, Width)


# ============================================================================
# MAIN CLASS
# ============================================================================

class ModelBenchmark:
    """
    Class để thực hiện benchmark (tính FLOPs, Parameters) cho mô hình PyTorch.
    
    Attributes:
        model: Mô hình PyTorch.
        device: Device (cpu/cuda).
        model_name: Tên mô hình.
        input_shape: Kích thước input.
        results: Kết quả benchmark cuối cùng.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str = "Model",
        device: Optional[torch.device] = None,
        input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE,
    ):
        """
        Khởi tạo ModelBenchmark.
        
        Args:
            model: Mô hình PyTorch.
            model_name: Tên mô hình để hiển thị.
            device: Device (auto-detect nếu None).
            input_shape: Kích thước input (B, C, H, W).
        """
        self.model = model
        self.model_name = model_name
        self.device = device or self._get_device()
        self.input_shape = input_shape
        self.results: Optional[Dict[str, float]] = None
        
        # Chuyển model sang device
        self.model.to(self.device).eval()
    
    @staticmethod
    def _get_device(use_cuda: bool = True) -> torch.device:
        """
        Lấy device phù hợp (GPU hoặc CPU).
        
        Args:
            use_cuda: Nếu True, sử dụng GPU nếu có sẵn.
            
        Returns:
            torch.device: Device object.
        """
        if use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def load_checkpoint(self, checkpoint_path: Path) -> "ModelBenchmark":
        """
        Load trọng số từ checkpoint vào mô hình.
        
        Args:
            checkpoint_path: Đường dẫn đến checkpoint file.
            
        Returns:
            Self để hỗ trợ method chaining.
            
        Raises:
            FileNotFoundError: Nếu checkpoint không tồn tại.
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"❌ Checkpoint không tìm thấy: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Xác định key state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        print(f"✅ Đã load checkpoint từ: {checkpoint_path}")
        
        return self
    
    def calculate(self) -> Dict[str, float]:
        """
        Tính FLOPs và Parameters của mô hình.
        
        Returns:
            Dict chứa:
                - "params_m": Số parameters (triệu)
                - "flops_g": Số FLOPs (tỷ)
                - "params_raw": Số parameters (raw)
                - "flops_raw": Số FLOPs (raw)
                
        Raises:
            RuntimeError: Nếu có lỗi khi tính toán.
        """
        # Tạo input giả lập
        dummy_input = torch.randn(self.input_shape, device=self.device)
        
        try:
            with torch.no_grad():
                flops, params = profile(
                    self.model,
                    inputs=(dummy_input,),
                    verbose=False
                )
            
            # Chuyển đổi đơn vị
            params_m = params / UNIT_CONVERSION["params"]
            flops_g = flops / UNIT_CONVERSION["flops"]
            
            self.results = {
                "params_m": round(params_m, 2),
                "flops_g": round(flops_g, 2),
                "params_raw": params,
                "flops_raw": flops,
            }
            
            return self.results
            
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi khi tính FLOPs/Params: {str(e)}")
    
    def print_results(self) -> None:
        """
        In kết quả benchmark dưới dạng dễ đọc.
        
        Raises:
            RuntimeError: Nếu chưa tính toán kết quả.
        """
        if self.results is None:
            raise RuntimeError("❌ Chưa tính toán kết quả. Gọi .calculate() trước.")
        
        print("\n" + "=" * 60)
        print(f"📊 BENCHMARK RESULTS - {self.model_name}")
        print("=" * 60)
        print(f"   Device:     {self.device}")
        print(f"   Input:      {self.input_shape}")
        print(f"   Parameters: {self.results['params_m']:.2f} M")
        print(f"   FLOPs:      {self.results['flops_g']:.2f} G")
        print("=" * 60 + "\n")
    
    def run(self, verbose: bool = True) -> Dict[str, float]:
        """
        Thực hiện benchmark đầy đủ (tính + in kết quả).
        
        Args:
            verbose: In kết quả hay không.
            
        Returns:
            Dict chứa kết quả benchmark.
        """
        self.calculate()
        
        if verbose:
            self.print_results()
        
        return self.results
    
    def get_results(self) -> Optional[Dict[str, float]]:
        """
        Lấy kết quả benchmark hiện tại.
        
        Returns:
            Dict chứa kết quả hoặc None nếu chưa tính.
        """
        return self.results
    
    def set_input_shape(self, input_shape: Tuple[int, ...]) -> "ModelBenchmark":
        """
        Thay đổi kích thước input.
        
        Args:
            input_shape: Kích thước input mới.
            
        Returns:
            Self để hỗ trợ method chaining.
        """
        self.input_shape = input_shape
        self.results = None  # Reset results
        return self
    
    def __repr__(self) -> str:
        """String representation của object."""
        status = "✓ Calculated" if self.results else "○ Not calculated"
        return f"ModelBenchmark(name='{self.model_name}', device={self.device}, {status})"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Config
    MODEL_NAME = "mobilenetv3_small"
    MODEL_RUN_ID = "run_20251021-151012"
    NUM_CLASSES = 10
    
    print("🖥️  ModelBenchmark Example")
    print("-" * 60)
    
    # Cách 1: Tạo model đơn giản để demo
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(32, NUM_CLASSES)
    )
    
    # Khởi tạo benchmark
    benchmark = ModelBenchmark(
        model=model,
        model_name=MODEL_NAME,
        input_shape=(1, 3, 224, 224)
    )
    
    print(f"\n{benchmark}\n")
    
    # Cách 2: Load checkpoint (nếu có)
    # checkpoint_path = (
    #     Path(__file__).resolve().parents[2] 
    #     / "checkpoints" 
    #     / MODEL_NAME 
    #     / MODEL_RUN_ID 
    #     / "best_checkpoint.pth"
    # )
    # benchmark.load_checkpoint(checkpoint_path)
    
    # Thực hiện benchmark
    results = benchmark.run(verbose=True)
    
    # Hoặc tính từng bước
    # benchmark.calculate()
    # benchmark.print_results()
    # results = benchmark.get_results()
    
    # Đổi input shape và tính lại
    # benchmark.set_input_shape((1, 3, 512, 512)).run()
    
    print("✅ Example completed successfully!")