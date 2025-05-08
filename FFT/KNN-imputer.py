from sklearn.impute import KNNImputer

def df_val(df):
    df_zero_counts = (df == 0).sum(axis=1) # 0으로 이루어진 시퀀스 값 삭제
    df_beta = df[df_zero_counts < 1]
    time_columns = [col for col in df_beta.columns if "elcp_use_" in col]

    df_ente_list = df_beta['ente'].unique() # 기업의 수

    # 기업별로 나누어 데이터프레임 생성
    df_ente = []
    for i in range(len(df_ente_list)):
        df_ente.append(df_beta[df_beta['ente'] == df_ente_list[i]])

    df_ente_time = []
    for i in range(len(df_ente)):
        df_ente_ = df_ente[i]
        df_ente_time.append(df_ente_[time_columns])

    imputer = KNNImputer(n_neighbors=3, weights="uniform") # Imputer 객체 생성

    df_ente_imputed = [] # 각각의 데이터프레임에 대해 결측치 알고리즘을 적용한 데이터들을 저장할 리스트 생성
    for i in range(len(df_ente_time)):
        df_ente_imputed.append(imputer.fit_transform(df_ente_time[i]))

    df_ente_filterd = []
    for i in range(len(df_ente_imputed)):
        df_ente_filterd.append(pd.DataFrame(df_ente_imputed[i], columns=time_columns))

    df_ente_dd = []
    for i in range(len(df_ente_filterd)):
        df_ente_dd.append(df_ente_filterd[i].set_index(df_ente[i].index))
    
    for i in range(len(df_ente_dd)):
        df_ente_dd[i].insert(0, 'code_tite', df_ente[i]['kemc_oldx_code_tite'])
        df_ente_dd[i].insert(1, 'ente', df_ente[i]['ente'])
        df_ente_dd[i].insert(2, 'cntr_tp_name', df_ente[i]['cntr_tp_name'])
        df_ente_dd[i].insert(3, 'season', df_ente[i]['season_name'])
        df_ente_dd[i].insert(4, 'weekd_weekend', df_ente[i]['weekd_weekend_name'])

    df_omega = pd.concat(df_ente_dd)

    return df_omega