def atbgraph(gm, fake_input_tensor):
    from dlinfer.graph.dicp.dynamo_bridge.compile_fx import compile_fx

    return compile_fx(gm, fake_input_tensor, "atbgraph")
