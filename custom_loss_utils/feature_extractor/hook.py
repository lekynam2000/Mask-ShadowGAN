def hook_fn_fac(feature_obj,key):
    def hook(m,i,o):
        feature_obj[key]=o
        return hook
