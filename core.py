import numpy as np
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import tensorflow as tf
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd

class D2TModel:
    def __init__(self, config):
        self.config = config
        self.base_model = TFAutoModelForSeq2SeqLM.from_pretrained(self.config['model_name'])
        self.config = AutoConfig.from_pretrained(self.config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossEntropyWithLogits(from_logits=True)
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr'])
        self.scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<H>', '<R>', '<T>']})


    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        return lr * tf.math.exp(-0.1)

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            outputs = self.base_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], decoder_input_ids=targets['decoder_input_ids'])
            logits = outputs.logits
            loss = self.loss_fn(targets['decoder_input_ids'], logits)
        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        return loss

    def process_data(self):
        prefix = self.config['prefix']
        with open(self.config['source_data_path']) as f:
            lines = f.readlines()
            source = [prefix + ' ' + a.strip() for a in lines]
        with open(self.config['target_data_path']) as f:
            lines = f.readlines()
            target = [prefix + ' ' + a.strip() for a in lines]
        return source, target

    def encode(self, data):
        return self.tokenizer.batch_encode_plus(data, max_length=self.config['max_length'], padding='max_length', truncation=True, return_tensors='tf')

    def train(self, input_ids, target_ids):
        b = self.config['batch_size']
        loss = float('inf')
        for e in self.config['epochs']:
            for i in range(len(input_ids)-b):
                ip = input_ids[i:i+b]
                op = target_ids[i:i + b]
                loss = self.train_step(ip, op)
            print(loss)

if __name__=='__main__':
    pass


