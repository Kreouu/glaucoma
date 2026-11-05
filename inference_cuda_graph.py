"""Replay the unchanged six-model FP16 path; no image or prediction caching."""
import torch


class GraphPredictor:
    @torch.inference_mode()
    def __init__(self, predictor):
        torch.set_num_threads(4)
        self.predictor = predictor
        self.input = torch.zeros((1, 3, 512, 512), device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                predictor.logits(self.input).mean(0).sigmoid()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = predictor.logits(self.input).mean(0).sigmoid()

    @torch.inference_mode()
    def predict_one(self, path):
        tensor = self.predictor.predictors[0].preprocess(path).unsqueeze(0)
        self.input.copy_(tensor)
        self.graph.replay()
        value = self.output.item()
        assert 0 <= value <= 1
        return value, int(value >= .5)
