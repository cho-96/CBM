#================================================================================================
#--------------------------------------   Library Functions    ----------------------------------
#------------------------------------------------------------------------------------------------
#------- By : SafeEye (이광준, 임헌규)
#------- Last Updated : 2022.06.01
#================================================================================================
import pandas as pd
import numpy  as np

#------------------------------------------------------------------------------------------------
def getCorrelationMatrix(df, f_draw=True) :  #--------- Dataframe의 Correlation Matrix 구하기
#------------------------------------------------------------------------------------------------
    import seaborn as sns
    df_corr = df.corr()
    if not f_draw :
        return df_corr

    fig, ax = plt.subplots( figsize=(7,7) )

    # 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
    mask = np.zeros_like(df_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # 히트맵을 그린다
    sns.heatmap(df_corr, 
                cmap = 'RdYlBu_r', 
                annot = True,   # 실제 값을 표시한다
                mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
                linewidths=.5,  # 경계면 실선으로 구분하기
                cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
                vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
               )  
    plt.show()
    return df_corr


#------------------------------------------------------------------------------------------------
def normalizeDataframe(df, scaling='Z-Score') :   #---  Data Normalization : 3 Selectable Methods
#------------------------------------------------------------------------------------------------
	global pd
    if scaling=='Z-Score' :
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        

    if scaling=='MinMax' :
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    if scaling=='MaxAbs' :
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        
    scaler.fit(df)
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    return scaled_df
