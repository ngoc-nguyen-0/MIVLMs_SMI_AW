from llava.train.train import train

if __name__ == "__main__":
    
    from setproctitle import setproctitle
    setproctitle(f"Train model")
    train(attn_implementation="flash_attention_2")
