#%%
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#데이터를 로드하고 두 개의 클래스만 선택합니다.
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# 처음 40개 샘플을 제거하여 불균형한 클래스를 만듭니다.
features = features[40:,:]
target = target[40:]

# 타깃 벡터에서 0이 아닌 클래스는 모두 1로 만듭니다.
target = np.where((target == 0), 0, 1)

# 특성을 표준화합니다.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 서포트 벡터 분류기를 만듭니다.
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)

# 분류기를 훈련합니다.
model = svc.fit(features_standardized, target)
# %%
