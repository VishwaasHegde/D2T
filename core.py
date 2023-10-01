import numpy as np
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import tensorflow as tf
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
import json

class D2TModel:
    def __init__(self):
        with open('config.json', 'r') as f:
            config_params = json.load(f)

        self.config_params = config_params
        self.base_model = TFAutoModelForSeq2SeqLM.from_pretrained(config_params['model_name'], cache_dir=config_params['model_folder'])
        self.config = AutoConfig.from_pretrained(config_params['model_name'], cache_dir=config_params['config_cache'])
        self.tokenizer = AutoTokenizer.from_pretrained(config_params['model_name'], cache_dir=config_params['tokenizer_cache'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config_params['lr'])
        self.base_model.optimizer = self.optimizer
        self.scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        self.scheduler.set_model(self.base_model)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=config_params['model_folder'], monitor='val_loss', save_best_only=True, mode='min')
        self.model_checkpoint.set_model(self.base_model)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<H>', '<R>', '<T>']})


    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        return lr * tf.math.exp(-0.1)

    def train_step(self, input_ids, attention_mask, decoder_input_ids):
        with tf.GradientTape() as tape:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            loss = self.loss_fn(decoder_input_ids, logits)
        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        return loss

    def test_step(self, input_ids, attention_mask, decoder_input_ids):
        with tf.GradientTape() as tape:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            loss = self.loss_fn(decoder_input_ids, logits)
        return loss

    def validate(self):
        test_data_x, test_data_y = self.process_data('val')
        input_ids = self.encode(test_data_x)
        target_ids = self.encode(test_data_y)
        b = self.config_params['batch_size']
        loss = 0
        for i in range(len(test_data_x)-b):
            source_input_ids = input_ids['input_ids'][i:i+b]
            source_attention_mask = input_ids['attention_mask'][i:i + b]
            decoder_input_ids = target_ids['input_ids'][i:i + b]
            decoded_outputs = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
            loss += self.test_step(source_input_ids, source_attention_mask, decoder_input_ids)
        return loss/(len(test_data_x)-b)

    def process_data(self, task):
        prefix = self.config_params['prefix']
        with open(self.config_params[f'{task}_source_data_path']) as f:
            lines = f.readlines()
            source = [prefix + ' ' + a.strip() for a in lines[:100]]
        with open(self.config_params[f'{task}_target_data_path']) as f:
            lines = f.readlines()
            target = [a.strip() for a in lines[:100]]
        return source, target

    def generate(self, input_text):
        input_id = self.encode([input_text])


    def test(self):
        test_data_x, test_data_y = self.process_data('test')
        input_ids = self.encode(test_data_x)
        target_ids = self.encode(test_data_y)
        b = self.config_params['batch_size']
        loss = 0

        for i in range(len(test_data_x)-b):
            source_input_ids = input_ids['input_ids'][i:i+b]
            source_attention_mask = input_ids['attention_mask'][i:i + b]
            decoder_input_ids = target_ids['input_ids'][i:i + b]
            decoded_outputs = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
            loss += self.test_step(source_input_ids, source_attention_mask, decoder_input_ids)
        print(loss/(len(test_data_x)-b))

    def encode(self, data):
        return self.tokenizer.batch_encode_plus(data, max_length=self.config_params['max_length'], padding='max_length', truncation=True, return_tensors='tf')

    def train(self):
        train_data_x, train_data_y = self.process_data('train')
        input_ids = self.encode(train_data_x)
        target_ids = self.encode(train_data_y)

        b = self.config_params['batch_size']
        loss = 0
        for e in range(self.config_params['epochs']):
            for i in range(len(train_data_x)-b):
                source_input_ids = input_ids['input_ids'][i:i+b]
                source_attention_mask = input_ids['attention_mask'][i:i + b]
                decoder_input_ids = target_ids['input_ids'][i:i + b]
                loss = self.train_step(source_input_ids, source_attention_mask, decoder_input_ids)
            print(loss)
            val_loss = self.validate()
            self.scheduler.on_epoch_end(e+1, {'val_loss': val_loss})
            self.model_checkpoint.on_epoch_end(e + 1, {'val_loss': val_loss})

if __name__=='__main__':
    d2t_model = D2TModel()
    d2t_model.train()


