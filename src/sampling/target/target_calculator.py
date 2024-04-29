from abc import abstractmethod


class TargetCalculator:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj

    @abstractmethod
    def get_output(self, cm):
        pass
