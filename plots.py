if __name__ == "__main__":
    
    M = 25.					# JLA normalization 
    h = 0.7302 
    omega0_b = 0.022
    omega0_cdm = 0.048
    params = [M, h, omega0_b, omega0_cdm]
    
    # test_cc(params)
    # test_jla(params)
    # test_cmb(params)
    # test_bao(params)
    # test_wigglez(params)
    # test_fap(params)
    
    file = "/Users/fabio/code/fchibana/tachyons/config.yaml"
    
    config = load_config(file)
    print(config)