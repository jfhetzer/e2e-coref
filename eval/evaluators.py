from abc import ABC, abstractmethod


class MentionEvaluator(ABC):

    def __init__(self, name):
        self.name = name
        self._num_correct = 0
        self._num_gold = 0
        self._num_preds = 0

    def update(self, cand_starts, cand_ends, gold_starts, gold_ends, ment_starts, ment_ends, cand_scores, sent_len):
        # make parameters available
        self.num_words = sum(sent_len)
        self.gold_spans = set(zip(gold_starts.cpu().numpy(), gold_ends.cpu().numpy()))
        self.cand_starts = cand_starts.cpu().numpy()
        self.cand_ends = cand_ends.cpu().numpy()
        self.ment_starts = ment_starts.cpu().numpy()
        self.ment_ends = ment_ends.cpu().numpy()
        self.cand_scores = cand_scores.cpu().numpy()

        sorted_idx = cand_scores.argsort(descending=True).cpu().numpy()
        self.sorted_starts = self.cand_starts[sorted_idx]
        self.sorted_ends = self.cand_ends[sorted_idx]

        # calculate actual predictions
        preds = self.get_preds()

        # update counters
        self._num_correct += len(self.gold_spans & preds)
        self._num_gold += len(self.gold_spans)
        self._num_preds += len(preds)

    def print_result(self):
        # calculate metrics
        recall = self._num_correct / self._num_gold if self._num_gold else 0
        precision = self._num_correct / self._num_preds if self._num_preds else 0
        f1 = 2 * recall * precision / (precision + recall) if recall + precision else 0
        # print results
        print(f'F @ {self.name}: {f1:.2f}, ', end='')
        print(f'P @ {self.name}: {precision:.2f}, ', end='')
        print(f'R @ {self.name}: {recall:.2f}')

    @abstractmethod
    def get_preds(self):
        pass


class OracleEvaluator(MentionEvaluator):

    def get_preds(self):
        return set(zip(self.cand_starts, self.cand_ends)) & self.gold_spans


class ActualEvaluator(MentionEvaluator):

    def get_preds(self):
        return set(zip(self.ment_starts, self.ment_ends))


class ExactEvaluator(MentionEvaluator):

    def get_preds(self):
        pred_starts = self.sorted_starts[:len(self.gold_spans)]
        pred_ends = self.sorted_ends[:len(self.gold_spans)]
        return set(zip(pred_starts, pred_ends))


class ThresholdEvaluator(MentionEvaluator):

    def get_preds(self):
        ment_mask = self.cand_scores > 0
        pred_starts = self.cand_starts[ment_mask]
        pred_ends = self.cand_ends[ment_mask]
        return set(zip(pred_starts, pred_ends))


class RelativeEvaluator(MentionEvaluator):

    def __init__(self, name, k):
        super().__init__(name)
        self.k = k

    def get_preds(self):
        num_pred = int(self.k * self.num_words / 100)
        pred_starts = self.sorted_starts[:num_pred]
        pred_ends = self.sorted_ends[:num_pred]
        return set(zip(pred_starts, pred_ends))
