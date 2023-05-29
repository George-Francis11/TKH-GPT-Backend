import time
import gpt_2_simple as gpt2
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class GPT2:
    def __init__(self, model_name="124M", run_name="NewDatasetRun1"):
        self.model_name = model_name
        self.run_name = run_name
        self.prefix = "Q: "
        self.prediction = "A:"
        if not os.path.isdir(os.path.join("models", model_name)):
            print(f"Printing {model_name} GPT-2 model")
            gpt2.download_gpt2(model_name=self.model_name)
        else:
            print("Model was already downloaded")

    def clean_prefix(self, prefix:str):
        prefix = prefix.strip()
        prefix = prefix.lower()
        if prefix[-1] != "?":
            prefix = prefix + "?"
        if prefix[0:2] not in ["Q:","q:"]:
            prefix = "Q: " + prefix
        self.prefix = prefix
        return prefix

    def predict(self, prefix="what is tkh?"):
        self.clean_prefix(prefix)
        start = time.time()
        tf.reset_default_graph()
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name=self.run_name)

        try:
            self.prediction = gpt2.generate(
                sess,
                length=100,
                temperature=0.7,
                prefix=self.prefix,
                run_name=self.run_name,
                nsamples=1,
                batch_size=1,
                return_as_list=True,
                truncate="Q:",
                include_prefix=False
                )
            self.prediction[0] = self.prediction[0].replace("|||","")
            self.prediction[0] = self.prediction[0].replace("A:","")
        except:
            self.prediction = ["Could not find an answer to your question, I am sorry :'("]
        return self.prediction