from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig
import os

def check_config(data_config,model_config,training_config):
    data_config.max_src_len = model_config.max_src_len
    data_config.max_trg_len = model_config.max_trg_len
    data_config.min_trg_len = model_config.min_trg_len
    training_config.output_dir = os.path.join('../results/','test')
    data_config.use_image = model_config.use_image
    data_config.use_keyword = model_config.use_keyword
    if model_config.use_image or model_config.use_keyword:
        model_config.model_type = 'mm_transformer'
    else:
        model_config.model_type == 'transformer'
    return data_config,model_config,training_config


@dataclass
class DataConfig:
    use_cache:bool = False
    trg_lang:str = 'fr'
    test_year:str = '2017'

    dataset_prefix = '../data/'
    train_src_dir:str = '../data/train.lc.norm.tok.'+'en'
    train_trg_dir:str = 'train.lc.norm.tok.'

    dev_src_dir:str = '../data/val.lc.norm.tok.'+'en'
    dev_trg_dir:str = 'val.lc.norm.tok.'

    test_src_dir:str = ''
    test_trg_dir:str = ''   

    stop_word_dir:str = '../data/stopwords-en.txt'
    
    w:int = 8
    m:int = 5

    min_freq:int = 2

    ## no change
    use_image:bool=False
    use_keyword:bool=False

    def __post_init__(self):
        
        self.train_trg_dir = os.path.join(self.dataset_prefix,'train.lc.norm.tok.'+self.trg_lang)
        self.dev_trg_dir = os.path.join(self.dataset_prefix,"val.lc.norm.tok."+self.trg_lang)
        self.test_src_dir = os.path.join(self.dataset_prefix,'test_'+self.test_year+'_mscoco.lc.norm.tok.en')
        self.test_trg_dir = os.path.join(self.dataset_prefix,'test_'+self.test_year+'_mscoco.lc.norm.tok.'+self.trg_lang)


class ModelConfig(PretrainedConfig):
    model_type='marian'
    def __init__(
        self,
        model_type = 'mm_transformer', # transformer,mm_transformer
        vocab_size=10,
        max_position_embeddings=512,

        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        
        use_cache=True,
        is_encoder_decoder=True,
        
        activation_function="relu",
        d_model=512,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=3,
        scale_embedding=True,
        gradient_checkpointing=False,
        pad_token_id=1,
        eos_token_id=2,
        forced_eos_token_id=0,
        bos_token_id=3,

        use_image=False,
        use_resnet = False,
        use_keyword=True,

        max_src_len = 100,
        max_trg_len = 100,
        min_trg_len = 3,
        num_beams=5,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.use_resnet = use_resnet
        self.num_beams = num_beams
        self.max_trg_len = max_trg_len
        self.min_trg_len = min_trg_len
        self.max_src_len = max_src_len
        self.model_type=model_type
        self.vocab_size=vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.encoder_layers=encoder_layers
        self.encoder_ffn_dim=encoder_ffn_dim
        self.encoder_attention_heads=encoder_attention_heads
        self.decoder_layers=decoder_layers
        self.decoder_ffn_dim=decoder_ffn_dim
        self.decoder_attention_heads=decoder_attention_heads
        self.encoder_layerdrop=encoder_layerdrop
        self.decoder_layerdrop=decoder_layerdrop
        self.use_cache=use_cache
        self.is_encoder_decoder=is_encoder_decoder
        self.activation_function=activation_function
        self.d_model=d_model
        self.dropout=dropout
        self.attention_dropout=attention_dropout
        self.activation_dropout=activation_dropout
        self.init_std=init_std
        self.decoder_start_token_id=decoder_start_token_id
        self.scale_embedding=scale_embedding
        self.gradient_checkpointing=gradient_checkpointing
        self.pad_token_id=pad_token_id
        self.eos_token_id=eos_token_id
        self.forced_eos_token_id=forced_eos_token_id
        self.bos_token_id=bos_token_id
        self.use_image=use_image
        self.use_keyword=use_keyword

@dataclass
class TrainingConfig:

    output_dir: str = 'test'

    eval_steps: int = 2000
    start_eval_step: int = 5000
    
    logging_steps: int = 100
    
    lr_scheduler_type:str = 'linear'
    learning_rate:float = 2e-6
    weight_decay:float=0
    
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 128
    world_size:int = 1
    per_device_eval_batch_size: int = 150  # sentences
    gradient_accumulation_steps: int = 1

    max_grad_norm: float = 1.0
    num_warmup_steps: int = 1200

    seed: int = 42
    

