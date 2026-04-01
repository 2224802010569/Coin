from feature.processing.output.processing_out import ProcessingOutput

class ProcessingInput:
    def run(self, tf = "1d"):
        return ProcessingOutput().run(tf = tf)