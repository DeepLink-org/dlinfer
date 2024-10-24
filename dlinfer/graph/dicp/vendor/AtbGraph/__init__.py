def atbgraph(gm, fake_input_tensor):
    import torch._dynamo.config

    torch._dynamo.config.cache_size_limit = 256
    from dlinfer.graph.dicp.dynamo_bridge.compile_fx import compile_fx

    return compile_fx(gm, fake_input_tensor, "atbgraph")
