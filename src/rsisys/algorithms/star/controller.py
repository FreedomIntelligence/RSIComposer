"""Schedule STaR from matched Lab receipts and select a trained checkpoint."""

from rsisys.api import Decision, ExecutionStatus, LabRequest

from .protocol import curriculum


class StarController:
    def __init__(self, context, config):
        self.schedule = curriculum(config['rounds'], config['plain_rounds'])
        self.flow = self.operations()
        self.pending = None
        self.result = None

    def required_operations(self):
        return ('star.evaluate', 'star.collect', 'star.fit')

    def report(self):
        return self.result or {'method': 'star', 'status': 'running'}

    def operations(self):
        baseline_dev = yield "evaluate", {"split": "dev", "model": "base", "label": "baseline-dev"}, "评测基座验证集"
        baseline_test = yield "evaluate", {"split": "test", "model": "base", "label": "baseline-test"}, "评测基座测试集"
        best_model, best = "base", baseline_dev["quality_metrics"]
        previous, rounds = "base", []
        for number, rationalize, steps in self.schedule:
            collected = yield "collect", {"round": number, "model": previous,
                                          "rationalize": rationalize}, f"第 {number} 轮 · 生成和筛选推理"
            fitted = yield "fit", {"round": number, "dataset": collected["dataset"],
                                  "steps": steps}, f"第 {number} 轮 · 从基座训练"
            previous = fitted["model"]
            measured = yield "evaluate", {"split": "dev", "model": previous,
                                          "label": f"{previous}-dev"}, f"第 {number} 轮 · 验证新权重"
            metrics = measured["quality_metrics"]
            rounds.append({"round": number, "model": previous, "checkpoint": fitted["checkpoint"],
                           "collection": collected["collection"], "training": fitted["training_metrics"],
                           "development": metrics})
            if metrics["correct"] > best["correct"]:
                best_model, best = previous, metrics
        selected = max(rounds, key=lambda row: row["development"]["correct"])
        trained_test = yield "evaluate", {"split": "test", "model": selected["model"],
                                        "label": "selected-test"}, "评测验证集选中的训练权重"
        self.result = {"status": "completed", "rounds": rounds, "best_including_base": best_model,
                       "baseline_dev": baseline_dev["quality_metrics"],
                       "baseline_test": baseline_test["quality_metrics"], "selected": selected["model"],
                       "selected_test": trained_test["quality_metrics"]}

    def decide(self, window):
        feedback = window.feedback
        try:
            if self.pending is None:
                name, request, reason = next(self.flow)
            else:
                if (window.last_operation != self.pending or feedback is None or feedback.status != ExecutionStatus.COMPLETED
                        or feedback.data["request"] != self.pending.arguments):
                    raise RuntimeError("STaR feedback differs from its request")
                name, request, reason = self.flow.send(feedback.data)
        except StopIteration:
            return Decision("finish", reason="STaR training and held-out evaluation completed")
        self.pending = LabRequest(f"star.{name}", request)
        return Decision("invoke", operation=self.pending, reason=reason)

