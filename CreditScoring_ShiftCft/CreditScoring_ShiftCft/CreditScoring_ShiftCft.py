
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mth
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import  sklearn.preprocessing as pr
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
import re as reg
import pylab
from collections import Counter
def get_sex(name, fam):
     sex = []
     for i, j in zip(name, fam):
        if (reg.search('.$', i).string[-1]=='а' or (reg.search('.$', i).string[-1]=='я')) and (reg.search('.$', j).string[-1]=='а' or (reg.search('.$', j).string[-1]=='я')):
           sex.append(0) #если женщина
        else: 
           sex.append(1)
     return sex
def get_phone_number(col):
    phone = pd.DataFrame({'Phone': {i: i[:9] for i in col}})
    un_phone = pd.DataFrame({'Phone':sorted(phone['Phone'].unique()),
                            'Unique_p':phone.groupby('Phone').size()})
    return un_phone

def col_to_binary(col):
    new_uniq = []
    for i in col:
        if i == 1 :
           new_uniq.append(0)
        else:
           new_uniq.append(1)
    return new_uniq
def col_to_binary_1(col):
    new_uniq = []
    for i in col:
        if i > 300 :
           new_uniq.append(0)
        else:
           new_uniq.append(1)
    return new_uniq
def cv_grid_search(scaledX, Y):
    #    нужно для подбора параметров
    #расскомментить, если требуется
    #param_test1 = {'n_estimators':range(10,100,1)}
    #param_test2 = {'max_depth': range(3,10,1)} 
    #param_test2 = {'alpha': np.linspace(1e-6, 1e-1, 10)} 
    param_test3 = {'learning_rate': np.linspace(0.2,1.,50)} 
    #gsearch1 = GridSearchCV(estimator = ensemble.GradientBoostingClassifier(max_depth = 4,
    # random_state=0, learning_rate=1.), 
    #param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    #gsearch1 = GridSearchCV(estimator = ensemble.GradientBoostingClassifier(n_estimators=49,
    # random_state=0, learning_rate=1.), 
    #param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1 = GridSearchCV(estimator = ensemble.GradientBoostingClassifier(n_estimators=49,
     max_depth = 3, random_state=0), 
    param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(scaledX,Y)
    print([gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_])

if __name__ == '__main__':
    head = ['Name', 'Age','Education','Children','Profession', 'Phone', 'Amount','Return_credit']
    data = pd.read_csv('ankety_final_learn.csv')
    data.columns = head
    name = data['Name'].apply(lambda x: x.split(' ')[0])
    fam = data['Name'].apply(lambda x: x.split(' ')[1])
    sex_t = get_sex(name, fam)
    sex_t = pd.DataFrame({'Sex':sex_t})
    data = data.join(sex_t)

    #Делаем датафрейм который находит для каждого имени количество записей в выборке
    un_name = pd.DataFrame({'Profession':sorted(data['Profession'].unique()),
                            'Unique':data.groupby('Profession').size()})
    #и джоиним ее с первоначальными данными
    data = data.join(un_name.set_index('Profession'), on='Profession')
    new_uniq = col_to_binary_1(data['Unique'])
    uniq = pd.DataFrame({'Unique_bin':new_uniq})
    data = data.join(uniq)
    #удаляем шум в обучающей выборке
    phone = data['Phone'].apply(lambda x: str(x))
    tmp = [i[:9] for i in phone]
    phone_new = pd.DataFrame({'Phone_new': tmp})
    data = data.join(phone_new)
    un_phone = pd.DataFrame({'Phone_new':sorted(data['Phone_new'].unique()),
                            'Unique_p':data.groupby('Phone_new').size()})
    data = data.join(un_phone.set_index('Phone_new'), on='Phone_new')
    new_uniq_phone = col_to_binary(data['Unique_p'])
    uniq_ph = pd.DataFrame({'Unique_ph':new_uniq_phone})
    data = data.join(uniq_ph)

    data = data.loc[ (data["Return_credit"] == 1) | (data["Return_credit"] == 0) ]
    #разделяем выборку на признаки и отклик
    Y =  data['Return_credit']
    X = data.drop(['Name','Profession', 'Phone','Return_credit', 'Phone_new', 'Unique', 'Unique_p'], axis=1)
    from sklearn.cross_validation import train_test_split
    #X, X_test, Y, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    #масштабируем признаки
    scaler = StandardScaler()
    scaler.fit(X)   
    scaledX = scaler.transform(X)
    #scl_test = scaler.transform(X_test)
    #cv_grid_search(scaledX, Y)
    #метод градиентного бустинга
    gbm = ensemble.GradientBoostingClassifier(n_estimators=65, learning_rate=1., max_depth=3, random_state=0)
    
    gbm.fit(scaledX, Y)
    #ошибка на обучающей выборке
    err_train = np.mean(Y != gbm.predict(scaledX))
    #работа с тестовой выборкой
    head_test = ['Name', 'Age','Education', 'Children','Profession', 'Phone', 'Amount']
    data_test = pd.read_csv('ankety_final_exam.csv')
    data_test.columns = head_test
    name_tst = data_test['Name'].apply(lambda x: x.split(' ')[0])
    fam_tst = data_test['Name'].apply(lambda x: x.split(' ')[1])
    sex_tst = get_sex(name_tst, fam_tst)
    sex_tst = pd.DataFrame({'Sex1':sex_tst})
    #Делаем датафрейм который находит для каждого имени количество записей в выборке
    #un_name1 = pd.DataFrame({'Name':sorted(data_test["Name"].unique()),
    #                        'Unique':data_test.groupby('Name').size()})
    ##и джоиним ее с первоначальными данными
    data_test = data_test.join(sex_tst)
    #new_uniq1 = col_to_binary_1(data_test['Amount'])
    #uniq1 = pd.DataFrame({'Unique_bin':new_uniq1})
    #data_test = data_test.join(uniq1)
    #data_test = data_test.join(un_name1.set_index("Name"), on='Name')

    #uniq_tst = col_to_binary(data_test['Unique'])
    #uniq_tst = pd.DataFrame({'Unique_bin':uniq_tst})
    #data_test = data_test.join(uniq_tst)
    un_name_t = pd.DataFrame({'Profession':sorted(data_test['Profession'].unique()),
                            'Unique':data_test.groupby('Profession').size()})
    #и джоиним ее с первоначальными данными
    data_test = data_test.join(un_name_t.set_index('Profession'), on='Profession')
    new_uniq_t = col_to_binary_1(data_test['Unique'])
    uniq_t = pd.DataFrame({'Unique_bin':new_uniq_t})
    data_test = data_test.join(uniq_t)
    #удаляем шум в обучающей выборке
    phone_t = data_test['Phone'].apply(lambda x: str(x))
    tmp_t = [i[:9] for i in phone_t]
    phone_new_t = pd.DataFrame({'Phone_new': tmp_t})
    data_test = data_test.join(phone_new_t)
    un_phone_t = pd.DataFrame({'Phone_new':sorted(data_test['Phone_new'].unique()),
                            'Unique_p':data_test.groupby('Phone_new').size()})
    data_test = data_test.join(un_phone_t.set_index('Phone_new'), on='Phone_new')
    new_uniq_phone_t = col_to_binary(data_test['Unique_p'])
    uniq_ph_t = pd.DataFrame({'Unique_ph':new_uniq_phone_t})
    data_test = data_test.join(uniq_ph_t)

    X_test = data_test.drop(['Name','Profession', 'Phone', 'Phone_new', 'Unique', 'Unique_p'], axis=1)
    #масштабируем признаки
    scaled_test = scaler.transform(X_test)
    #предсказываем отклик для тестовой выборки
    prediction = gbm.predict_proba(scaled_test)[:, 1]
    print(min(prediction), max(prediction))
    #записываем в файл
    result = pd.DataFrame(gbm.predict(scaled_test), columns=['otvet'])
    result.to_csv('result.csv', encoding='utf8')
