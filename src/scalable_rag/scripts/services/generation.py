from hydra import compose, initialize
from omegaconf import DictConfig


class GenerationService:
    def __init__(self):
        self.config: DictConfig = self.initialize_configs()

    def initialize_configs(self) -> DictConfig:
        with initialize(config_path="../../conf", job_name="ingestion"):
            return compose(config_name="config")

    def trigger(self): ...


# if __name__ == "__main__":
#     GenerationService().trigger()
