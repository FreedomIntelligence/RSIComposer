"""Optional observations: run the template with --probe <this-file>.

Keep probes with the method that owns them. They collect measured feedback;
controller decisions and Lab operations are unchanged.
"""


def register(observations):
    def quality(event):
        receipt = event.get('operation_result') or event.get('feedback') or {}
        return receipt.get('data', {}).get('quality_metrics') or None

    observations.register('quality', quality, on='feedback', every=1)
