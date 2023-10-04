from utils import *

def FMSA_attack(data_dir):
    steal_datas = FMSA_dataset(data_dir)
    E = Extractor()
    G = Generator()
    D = Discriminator()
    C = DNN_NIDS()
    meta_E = MAML(E)
    meta_G = MAML(G)
    meta_D = MAML(D)
    meta_C = MAML(C)
    fmsa_model = FMSA(meta_E, meta_G, meta_D, meta_C, data_dir)
    fmsa_loader = DataLoader(steal_datas, batch_size=5, shuffle=True)
    for data in fmsa_loader:
        support_x, support_y, query_x, query_y = data
        fmsa_model(support_x, support_y, query_x, query_y, meta_traing=True)
    return meta_E, meta_G, meta_D, meta_C, fmsa_model


if __name__ == "__main__":
    data_dir = 'data/CICdata.csv'
    meta_E, meta_G, meta_D, meta_C, fmsa_model = FMSA_attack(data_dir)
    