from transformers import AutoConfig, AutoModelForCausalLM

from ngram_transformers_wrapper.modeling_ngram import NgramConfig, NgramForCausalLM


def register_models() -> None:
    AutoConfig.register("ngram", NgramConfig)
    NgramForCausalLM.register_for_auto_class(AutoModelForCausalLM)
