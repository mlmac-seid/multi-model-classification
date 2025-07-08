# imports
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import seaborn as sns
import pandas as pd

# 3d figures
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# creating animations
import matplotlib.animation
from IPython.display import HTML

# styling additions
from IPython.display import HTML
style = '''
    <style>
        div.info{
            padding: 15px;
            border: 1px solid transparent;
            border-left: 5px solid #dfb5b4;
            border-color: transparent;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
        hr{
            border: 1px solid;
            border-radius: 5px;
        }
    </style>'''
HTML(style)

from sklearn import svm, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay, auc

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names[:2]

X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add 'noisy' features to make problem harder
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

classifier = svm.SVC(kernel="linear", C=0.01)

classifier.fit(X,y)

cm = confusion_matrix(y, classifier.predict(X))
cm

# display it on the subplot figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot();

RocCurveDisplay.from_estimator(classifier,X,y,name='ROC',lw=1);

cv = StratifiedKFold(n_splits=2)
cv.split(X,y)

for fold_num, (train_idx, test_idx) in enumerate(cv.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')
    print(f'train idices: {train_idx}')
    print()
    print(f'test idices: {test_idx}')

################################################
#                Setup figures
################################################
# NOTE: my default code has this working for 6 folds giving a 2x3 figure
fig_cm, ax_cm = plt.subplots(2,3,figsize=(10,8))

# setup ROC figure
fig, ax = plt.subplots(figsize=(10,8))
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic",
)

# setup mean curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# plot "no-skill"/"guessing" classifier
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

################################################
#               Cross-Validation
################################################
cv = StratifiedKFold(n_splits=6)

# actually perform the CV and loop through each fold
for fold_num, (train_idx, test_idx) in enumerate(cv.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')

    # fit classifier **on this folds data**
    classifier.fit(X[train_idx],y[train_idx])

    # predict on this folds "testing" set
    y_pred = classifier.predict(X[test_idx])

    # get this fold's confusion matrix
    cm = confusion_matrix(y[test_idx], classifier.predict(X[test_idx]))

    # display it on the subplot figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax_cm.flatten()[fold_num])
    ax_cm.flatten()[fold_num].set(title=f'CM for fold {fold_num+1}')

    # print this folds classification report
    print('Classification report:')
    print(classification_report(y[test_idx], y_pred, target_names=target_names))

    # build and display ROC curve for this fold
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test_idx],
        y[test_idx],
        name=f"ROC fold {fold_num+1}",
        alpha=0.4,
        lw=1,
        ax=ax,
    )

    # store information for the mean curve
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# now actually plot MEAN ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=f"Mean ROC (AUC = {mean_auc:0.2f})",
    lw=2,
    alpha=0.8,
)

ax.legend(loc="lower right");

from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y=True)
bc_df = load_breast_cancer(as_frame=True).data
bc_df_names = load_breast_cancer()
bc_target = bc_df_names.target_names
X.shape,y.shape

bc_df

# CLASSIFIER 1: SVM
classifier1 = svm.SVC(kernel="linear", C=0.01)
classifier1.fit(X,y)

cm1 = confusion_matrix(y, classifier1.predict(X))
cm1

# display it on the subplot figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=bc_target)
disp.plot();

RocCurveDisplay.from_estimator(classifier1,X,y,name='ROC',lw=1);

cv1 = StratifiedKFold(n_splits=2)
cv1.split(X,y)

for fold_num, (train_idx, test_idx) in enumerate(cv1.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')
    print(f'train idices: {train_idx}')
    print()
    print(f'test idices: {test_idx}')

################################################
#                Setup figures
################################################
# NOTE: my default code has this working for 6 folds giving a 2x3 figure
fig_cm, ax_cm = plt.subplots(2,3,figsize=(10,8))

# setup ROC figure
fig, ax = plt.subplots(figsize=(10,8))
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic",
)

# setup mean curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# plot "no-skill"/"guessing" classifier
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

################################################
#               Cross-Validation
################################################
cv1 = StratifiedKFold(n_splits=6)

# actually perform the CV and loop through each fold
for fold_num, (train_idx, test_idx) in enumerate(cv1.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')

    # fit classifier **on this folds data**
    classifier1.fit(X[train_idx],y[train_idx])

    # predict on this folds "testing" set
    y_pred = classifier1.predict(X[test_idx])

    # get this fold's confusion matrix
    cm1 = confusion_matrix(y[test_idx], classifier1.predict(X[test_idx]))

    # display it on the subplot figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=bc_target)
    disp.plot(ax=ax_cm.flatten()[fold_num])
    ax_cm.flatten()[fold_num].set(title=f'CM for fold {fold_num+1}')

    # print this folds classification report
    print('Classification report:')
    print(classification_report(y[test_idx], y_pred, target_names=bc_target))

    # build and display ROC curve for this fold
    viz = RocCurveDisplay.from_estimator(
        classifier1,
        X[test_idx],
        y[test_idx],
        name=f"ROC fold {fold_num+1}",
        alpha=0.4,
        lw=1,
        ax=ax,
    )

    # store information for the mean curve
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# now actually plot MEAN ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=f"Mean ROC (AUC = {mean_auc:0.2f})",
    lw=2,
    alpha=0.8,
)

ax.legend(loc="lower right");

# CLASSIFIER 2: LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression()

classifier2.fit(X, y)

cm2 = confusion_matrix(y, classifier2.predict(X))
cm2

# display it on the subplot figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=bc_target)
disp.plot();

RocCurveDisplay.from_estimator(classifier2,X,y,name='ROC',lw=1);

cv2 = StratifiedKFold(n_splits=2)
cv2.split(X,y)

for fold_num, (train_idx, test_idx) in enumerate(cv2.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')
    print(f'train idices: {train_idx}')
    print()
    print(f'test idices: {test_idx}')

################################################
#                Setup figures
################################################
# NOTE: my default code has this working for 6 folds giving a 2x3 figure
fig_cm, ax_cm = plt.subplots(2,3,figsize=(10,8))

# setup ROC figure
fig, ax = plt.subplots(figsize=(10,8))
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic",
)

# setup mean curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# plot "no-skill"/"guessing" classifier
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

################################################
#               Cross-Validation
################################################
cv2 = StratifiedKFold(n_splits=6)

# actually perform the CV and loop through each fold
for fold_num, (train_idx, test_idx) in enumerate(cv2.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')

    # fit classifier **on this folds data**
    classifier2.fit(X[train_idx],y[train_idx])

    # predict on this folds "testing" set
    y_pred = classifier2.predict(X[test_idx])

    # get this fold's confusion matrix
    cm2 = confusion_matrix(y[test_idx], classifier2.predict(X[test_idx]))

    # display it on the subplot figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=bc_target)
    disp.plot(ax=ax_cm.flatten()[fold_num])
    ax_cm.flatten()[fold_num].set(title=f'CM for fold {fold_num+1}')

    # print this folds classification report
    print('Classification report:')
    print(classification_report(y[test_idx], y_pred, target_names=bc_target))

    # build and display ROC curve for this fold
    viz = RocCurveDisplay.from_estimator(
        classifier2,
        X[test_idx],
        y[test_idx],
        name=f"ROC fold {fold_num+1}",
        alpha=0.4,
        lw=1,
        ax=ax,
    )

    # store information for the mean curve
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# now actually plot MEAN ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=f"Mean ROC (AUC = {mean_auc:0.2f})",
    lw=2,
    alpha=0.8,
)

ax.legend(loc="lower right");

# CLASSIFIER 3: KNN
from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors=3)

classifier3.fit(X,y)

cm3 = confusion_matrix(y, classifier3.predict(X))
cm3

# display it on the subplot figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=bc_target)
disp.plot();

RocCurveDisplay.from_estimator(classifier3,X,y,name='ROC',lw=1);

cv3 = StratifiedKFold(n_splits=2)
cv3.split(X,y)

for fold_num, (train_idx, test_idx) in enumerate(cv3.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')
    print(f'train idices: {train_idx}')
    print()
    print(f'test idices: {test_idx}')

################################################
#                Setup figures
################################################
# NOTE: my default code has this working for 6 folds giving a 2x3 figure
fig_cm, ax_cm = plt.subplots(2,3,figsize=(10,8))

# setup ROC figure
fig, ax = plt.subplots(figsize=(10,8))
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver Operating Characteristic",
)

# setup mean curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# plot "no-skill"/"guessing" classifier
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

################################################
#               Cross-Validation
################################################
cv3 = StratifiedKFold(n_splits=6)

# actually perform the CV and loop through each fold
for fold_num, (train_idx, test_idx) in enumerate(cv3.split(X,y)):
    print('=====================================================')
    print(f"CV fold #{fold_num+1}")
    print('=====================================================')

    # fit classifier **on this folds data**
    classifier3.fit(X[train_idx],y[train_idx])

    # predict on this folds "testing" set
    y_pred = classifier3.predict(X[test_idx])

    # get this fold's confusion matrix
    cm3 = confusion_matrix(y[test_idx], classifier3.predict(X[test_idx]))

    # display it on the subplot figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=bc_target)
    disp.plot(ax=ax_cm.flatten()[fold_num])
    ax_cm.flatten()[fold_num].set(title=f'CM for fold {fold_num+1}')

    # print this folds classification report
    print('Classification report:')
    print(classification_report(y[test_idx], y_pred, target_names=bc_target))

    # build and display ROC curve for this fold
    viz = RocCurveDisplay.from_estimator(
        classifier3,
        X[test_idx],
        y[test_idx],
        name=f"ROC fold {fold_num+1}",
        alpha=0.4,
        lw=1,
        ax=ax,
    )

    # store information for the mean curve
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# now actually plot MEAN ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=f"Mean ROC (AUC = {mean_auc:0.2f})",
    lw=2,
    alpha=0.8,
)

ax.legend(loc="lower right");
