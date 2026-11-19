"""Deterministic STaR execution fixture; produces no trained model weights."""
import json
from pathlib import Path


class FixtureStarEngine:
    simulated = True
    name = 'star-protocol-fixture/v1'

    def generate(self, items, model, *, count, temperature, hinted):
        return {item['id']: ['A fixture reasoning step.\n' + item['answer']
                if sample % 2 == 0 or hinted else 'invalid fixture response'
                for sample in range(count)] for item in items}

    def train(self, parent, dataset, checkpoint, *, steps):
        checkpoint.mkdir()
        (checkpoint / 'model.safetensors.index.json').write_text(json.dumps({
            'simulated': True, 'fixture': self.name, 'weights_produced': False}))
        return {'optimizer_steps': 0, 'requested_steps': steps,
                'train_rows': len(Path(dataset).read_text().splitlines())}
