# This is the helper function.



def dataProcess(df):
    import pandas as pd
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    from scipy.stats import skew
    from scipy import stats
    from sklearn.preprocessing import normalize
    plt.style.use('ggplot')

    train = df

    train.drop(train[(train['OverallQual'] < 5) & (
        train['SalePrice'] > 200000)].index, inplace=True)
    train.drop(train[(train['GrLivArea'] > 4000) & (
        train['SalePrice'] < 300000)].index, inplace=True)

    # BASEMENT
    train.BsmtCond = train.BsmtCond.fillna(0)
    train.BsmtExposure = train.BsmtExposure.fillna("None")
    train.BsmtQual = train.BsmtQual.fillna(0)
    train.BsmtFinType1 = train.BsmtFinType1.fillna("None")
    train.BsmtFinType2 = train.BsmtFinType2.fillna("None")
    train.BsmtFullBath = train.BsmtFullBath.fillna(0).astype(int)
    train.BsmtHalfBath = train.BsmtHalfBath.fillna(0).astype(int)
    train.BsmtUnfSF = train.BsmtUnfSF.fillna(0).astype(int)
    train.BsmtFinSF1 = train.BsmtFinSF1.fillna(0).astype(int)
    train.BsmtFinSF2 = train.BsmtFinSF2.fillna(0).astype(int)
    train.BsmtUnfSF = train.BsmtUnfSF.fillna(0).astype(int)
    train.TotalBsmtSF = train.TotalBsmtSF.fillna(0).astype(int)

    # LANDSLOPE
    train.LandSlope = train.LandSlope.fillna("None")

    # FIREPLACE
    train.FireplaceQu = train.FireplaceQu.fillna(0)

    # FENCE
    train.Fence = train.Fence.fillna("None")

    # FENCE
    train.PoolQC = train.PoolQC.fillna("None")

    # ALLEY
    train.Alley = train.Alley.fillna("None")

    # GARAGE
    train.GarageArea = train.GarageArea.fillna(0).astype(int)
    train.GarageFinish = train.GarageFinish.fillna("None")
    train.GarageType = train.GarageType.fillna("None")
    train.GarageQual = train.GarageQual.fillna(0)
    train.GarageCond = train.GarageCond.fillna(0)
    train.GarageCars = train.GarageCars.fillna(0).astype(int)
    train.GarageYrBlt = train.GarageYrBlt.fillna(train.YearBuilt).astype(int)

    # MISC FEATURE
    train.MiscFeature = train.MiscFeature.fillna("None")

    # MASONARY
    train.MasVnrType = train.MasVnrType.fillna("None")

    # ZONING - Most common
    train.MSZoning = train.MSZoning.fillna(train.MSZoning.mode()[0])

    # UTILITIES - Most common
    train.Utilities = train.Utilities.fillna(train.Utilities.mode()[0])

    # FUNCTIONAL - Most common
    train.Functional = train.Functional.fillna(train.Functional.mode()[0])

    # EXTERIOR - Most common - STRING
    train.Exterior1st = train.Exterior1st.fillna(train.Exterior1st.mode()[0])
    train.Exterior2nd = train.Exterior2nd.fillna(train.Exterior2nd.mode()[0])

    # ELECTRICAL - Most common
    train.Electrical = train.Electrical.fillna(train.Electrical.mode()[0])

    # KITCHEN QUALITY - Most common
    train.KitchenQual = train.KitchenQual.fillna("None")

    # SALE TYPE - Most common
    train.SaleType = train.SaleType.fillna(train.SaleType.mode()[0])

    # LOT FRONTAGE - Median - CAN BE PLAYED AROUND LATER
    train.LotFrontage = train.LotFrontage.fillna(train.LotFrontage.median())

    # Converting columns that should be strings from numericals
    train['MSSubClass'] = train['MSSubClass'].apply(str)
    train['YrSold'] = train['YrSold'].astype(str)
    train['MoSold'] = train['MoSold'].astype(str)

    # Simplifying Some Categorical Features
    mymap = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    train["ExterQual"] = train["ExterQual"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["ExterCond"] = train["ExterCond"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["BsmtQual"] = train["BsmtQual"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["BsmtCond"] = train["BsmtCond"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["HeatingQC"] = train["HeatingQC"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["PoolQC"] = train["PoolQC"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train['KitchenQual'] = train['KitchenQual'] = train["KitchenQual"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["FireplaceQu"] = train["FireplaceQu"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["GarageQual"] = train["GarageQual"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)
    train["GarageCond"] = train["GarageCond"].apply(
        lambda s: mymap.get(s) if s in mymap else s).astype(int)

    for col in ['BsmtFinType1', 'BsmtFinType2']:
        pd.Categorical(train[col], categories=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ'])

        pd.Categorical(train['LandSlope'], categories=[
                       'Gtl', 'Mod', 'Sev', 'None'])
        pd.Categorical(train['BsmtExposure'], categories=[
                       'Gd', 'Av', 'Mn', 'No', 'None'])

    # Further Simplification on Some Categorical Features
    train["Simplified_ExterQual"] = train.ExterQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_ExterCond"] = train.ExterCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_BsmtQual"] = train.BsmtQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_BsmtCond"] = train.BsmtCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_HeatingQC"] = train.HeatingQC.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_PoolQC"] = train.PoolQC.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_KitchenQual"] = train.KitchenQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_FireplaceQu"] = train.FireplaceQu.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_GarageQual"] = train.GarageQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_GarageCond"] = train.GarageCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    train["Simplified_OverallQual"] = train.OverallQual.replace({1: 1, 2: 1, 3: 1,
                                                                 4: 2, 5: 2, 6: 2,
                                                                 7: 3, 8: 3, 9: 3, 10: 3
                                                                 })
    train["Simplified_OverallCond"] = train.OverallCond.replace({1: 1, 2: 1, 3: 1,
                                                                 4: 2, 5: 2, 6: 2,
                                                                 7: 3, 8: 3, 9: 3, 10: 3
                                                                 })

    # Feature binaries
    train['CentralAir'] = train['CentralAir'].apply(
        lambda x: 0 if x == 'N' else 1)
    train['Street'] = train['Street'].apply(lambda x: 0 if x == 'Pave' else 1)
    train['PavedDrive'] = train['PavedDrive'].apply(
        lambda x: 0 if x == 'Y' else 1)
    train['LotShape'] = train['LotShape'].apply(
        lambda x: 1 if x == 'Reg' else 0)
    train['CentralAir'] = train['CentralAir'].apply(
        lambda x: 1 if x == "Y" else 0)
    train['Electrical'] = train['Electrical'].apply(
        lambda x: 1 if x == " Sbrkr" else 0)
    train['BsmtFullBath'] = train['BsmtFullBath'].apply(
        lambda x: 1 if x > 0 else 0)
    train['HalfBath'] = train['HalfBath'].apply(lambda x: 1 if x > 0 else 0)
    train['Fireplaces'] = train['Fireplaces'].apply(
        lambda x: 1 if x > 0 else 0)
    train['Functional'] = train['Functional'].apply(
        lambda x: 1 if x == 'Typ' else 0)
    train['PoolArea'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    train['WoodDeckSF'] = train['WoodDeckSF'].apply(
        lambda x: 1 if x > 0 else 0)
    train['ScreenPorch'] = train['ScreenPorch'].apply(
        lambda x: 1 if x > 0 else 0)
    train['OpenPorchSF'] = train['OpenPorchSF'].apply(
        lambda x: 1 if x > 0 else 0)
    train['EnclosedPorch'] = train['EnclosedPorch'].apply(
        lambda x: 1 if x > 0 else 0)
    train['3SsnPorch'] = train['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)

    # New Features
    train['Years_Since_Remodel'] = train['YrSold'].astype(
        int) - train['YearRemodAdd'].astype(int)
    train['Total_Home_Quality'] = train['OverallQual'] * train['OverallCond']
    train['Total_Exterior_Quality'] = train['ExterQual'] * train['ExterCond']
    train['Total_Kitchen_Quality'] = train['KitchenQual'] * train['KitchenAbvGr']
    train['Total_Fireplace_Quality'] = train['FireplaceQu'] * train['Fireplaces']
    train['Total_Garage_Quality'] = train['GarageQual'] * train['GarageArea']
    train['Total_Pool_Quality'] = train['PoolArea'] * train['PoolQC']
    train['Total_ Bath'] = train['BsmtFullBath'] + \
        (.5 * train['BsmtHalfBath']) + \
        train['FullBath'] + (.5 * train['HalfBath'])
    train['Total_Porch_SF'] = train['OpenPorchSF'] + \
        train["EnclosedPorch"] + train["3SsnPorch"] + train["ScreenPorch"]
    train['Total_SF'] = train["GrLivArea"] + train["TotalBsmtSF"]
    train["Total_FL_SF"] = train["1stFlrSF"] + train["2ndFlrSF"]
    train['Total_Interior_SF'] = train['TotalBsmtSF'] + \
        train['1stFlrSF'] + train['2ndFlrSF']

    train["Simple_Home_Quality"] = train["Simplified_OverallQual"] * \
        train["Simplified_OverallCond"]
    train["Simple_Exterior_Quality"] = train["Simplified_ExterQual"] * \
        train["Simplified_ExterCond"]
    train["Simple_Pool_Quality"] = train["PoolArea"] * \
        train["Simplified_PoolQC"]
    train["Simple_Garage_Quality"] = train["GarageArea"] * \
        train["Simplified_GarageQual"]
    train["Simple_Fireplace_Quality"] = train["Fireplaces"] * \
        train["Simplified_FireplaceQu"]
    train["Simple_Kitchen_Quality"] = train["KitchenAbvGr"] * \
        train["Simplified_KitchenQual"]


    return train