# This is the helper function.

# SaleCondition is categorical instead of being replaced with 1 if partially built house and 0 otherwise

def dataProcess(dataframe):
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

    try:
        dataframe.drop(dataframe[(dataframe['OverallQual'] < 5) & (
        dataframe['SalePrice'] > 200000)].index, inplace=True)

        dataframe.drop(dataframe[(dataframe['GrLivArea'] > 4000) & (
        dataframe['SalePrice'] < 300000) & (dataframe['SalePrice'] > 0)].index, inplace=True)

    except:
        pass

    # dataframe.drop(['Id'], axis=1, inplace=True)

    # BASEMENT
    dataframe.BsmtCond = dataframe.BsmtCond.fillna(0)
    dataframe.BsmtExposure = dataframe.BsmtExposure.fillna("None")
    dataframe.BsmtQual = dataframe.BsmtQual.fillna(0)
    dataframe.BsmtFinType1 = dataframe.BsmtFinType1.fillna("None")
    dataframe.BsmtFinType2 = dataframe.BsmtFinType2.fillna("None")
    dataframe.BsmtFullBath = dataframe.BsmtFullBath.fillna(0).astype(int)
    dataframe.BsmtHalfBath = dataframe.BsmtHalfBath.fillna(0).astype(int)
    dataframe.BsmtUnfSF = dataframe.BsmtUnfSF.fillna(0).astype(int)
    dataframe.BsmtFinSF1 = dataframe.BsmtFinSF1.fillna(0).astype(int)
    dataframe.BsmtFinSF2 = dataframe.BsmtFinSF2.fillna(0).astype(int)
    dataframe.BsmtUnfSF = dataframe.BsmtUnfSF.fillna(0).astype(int)
    dataframe.TotalBsmtSF = dataframe.TotalBsmtSF.fillna(0).astype(int)

    # LANDSLOPE
    dataframe.LandSlope = dataframe.LandSlope.fillna("None")

    # FIREPLACE
    dataframe.FireplaceQu = dataframe.FireplaceQu.fillna(0)

    # FENCE
    dataframe.Fence = dataframe.Fence.fillna("None")

    # POOLQC
    dataframe.PoolQC = dataframe.PoolQC.fillna("None")

    # ALLEY
    dataframe.Alley = dataframe.Alley.fillna("None")

    # GARAGE
    dataframe.GarageArea = dataframe.GarageArea.fillna(0).astype(int)
    dataframe.GarageFinish = dataframe.GarageFinish.fillna("None")
    dataframe.GarageType = dataframe.GarageType.fillna("None")
    dataframe.GarageQual = dataframe.GarageQual.fillna(0)
    dataframe.GarageCond = dataframe.GarageCond.fillna(0)
    dataframe.GarageCars = dataframe.GarageCars.fillna(0).astype(int)
    dataframe.GarageYrBlt = dataframe.GarageYrBlt.fillna(dataframe.YearBuilt).astype(int)

    # MISC FEATURE
    dataframe.MiscFeature = dataframe.MiscFeature.fillna("None")

    # MASONARY
    dataframe.MasVnrType = dataframe.MasVnrType.fillna("None")
    dataframe.MasVnrArea = dataframe.MasVnrArea.fillna(0)
    

    # ZONING - Most common
    dataframe.MSZoning = dataframe.MSZoning.fillna(dataframe.MSZoning.mode()[0])

    # UTILITIES - Most common
    dataframe.Utilities = dataframe.Utilities.fillna(dataframe.Utilities.mode()[0])

    # FUNCTIONAL - Most common
    dataframe.Functional = dataframe.Functional.fillna(dataframe.Functional.mode()[0])

    # EXTERIOR - Most common - STRING
    dataframe.Exterior1st = dataframe.Exterior1st.fillna(dataframe.Exterior1st.mode()[0])
    dataframe.Exterior2nd = dataframe.Exterior2nd.fillna(dataframe.Exterior2nd.mode()[0])

    # ELECTRICAL - Most common
    dataframe.Electrical = dataframe.Electrical.fillna(dataframe.Electrical.mode()[0])

    # KITCHEN QUALITY - Most common
    dataframe.KitchenQual = dataframe.KitchenQual.fillna("None")

    # SALE TYPE - Most common
    dataframe.SaleType = dataframe.SaleType.fillna(dataframe.SaleType.mode()[0])

    # LOT FRONTAGE - Median - CAN BE PLAYED AROUND LATER
    dataframe.LotFrontage = dataframe.LotFrontage.fillna(dataframe.LotFrontage.median())

    # Converting columns that should be strings from numericals
    dataframe['MSSubClass'] = dataframe['MSSubClass'].apply(str)
    dataframe['YrSold'] = dataframe['YrSold'].astype(str)
    dataframe['MoSold'] = dataframe['MoSold'].astype(str)

    # # Simplifying Some Categorical Features
    # mymap = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    # dataframe["ExterQual"] = dataframe["ExterQual"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["ExterCond"] = dataframe["ExterCond"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["BsmtQual"] = dataframe["BsmtQual"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["BsmtCond"] = dataframe["BsmtCond"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["HeatingQC"] = dataframe["HeatingQC"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["PoolQC"] = dataframe["PoolQC"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe['KitchenQual'] = dataframe["KitchenQual"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["FireplaceQu"] = dataframe["FireplaceQu"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["GarageQual"] = dataframe["GarageQual"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    # dataframe["GarageCond"] = dataframe["GarageCond"].apply(
    #     lambda s: mymap.get(s) if s in mymap else s).astype(int)
    
    # # PavedDrive
    # dataframe["PavedDrive"] = dataframe.PavedDrive.replace(
    #     {'N':'0', 'P':'1', 'Y':'2'}).astype(int)
    
    
    # Utilities. Drop column.
    dataframe = dataframe.drop(['Utilities'], axis=1)

    # for col in ['BsmtFinType1', 'BsmtFinType2']:                                            ########################## Shouldn't this be dataframe[col] = .... ? ###############
    #     pd.Categorical(dataframe[col], categories=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ'])

    # pd.Categorical(dataframe['LandSlope'], categories=[
    #                    'Gtl', 'Mod', 'Sev', 'None'])
    # pd.Categorical(dataframe['BsmtExposure'], categories=[
    #                    'Gd', 'Av', 'Mn', 'No', 'None'])

    # # Further Simplification on Some Categorical Features
    # dataframe["Simplified_ExterQual"] = dataframe.ExterQual.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_ExterCond"] = dataframe.ExterCond.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_BsmtQual"] = dataframe.BsmtQual.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_BsmtCond"] = dataframe.BsmtCond.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_HeatingQC"] = dataframe.HeatingQC.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_PoolQC"] = dataframe.PoolQC.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_KitchenQual"] = dataframe.KitchenQual.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_FireplaceQu"] = dataframe.FireplaceQu.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_GarageQual"] = dataframe.GarageQual.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_GarageCond"] = dataframe.GarageCond.replace(
    #     {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    # dataframe["Simplified_OverallQual"] = dataframe.OverallQual.replace({1: 1, 2: 1, 3: 1,
    #                                                              4: 2, 5: 2, 6: 2,
    #                                                              7: 3, 8: 3, 9: 3, 10: 3
    #                                                              })
    # dataframe["Simplified_OverallCond"] = dataframe.OverallCond.replace({1: 1, 2: 1, 3: 1,
    #                                                              4: 2, 5: 2, 6: 2,
    #                                                              7: 3, 8: 3, 9: 3, 10: 3
    #                                                              })

    # # Feature binaries
    # dataframe['CentralAir'] = dataframe['CentralAir'].apply(
    #     lambda x: 0 if x == 'N' else 1)
    # dataframe['Street'] = dataframe['Street'].apply(lambda x: 0 if x == 'Pave' else 1)
    # dataframe['LotShape'] = dataframe['LotShape'].apply(
    #     lambda x: 1 if x == 'Reg' else 0)
    # dataframe['CentralAir'] = dataframe['CentralAir'].apply(
    #     lambda x: 1 if x == "Y" else 0)
    # dataframe['Electrical'] = dataframe['Electrical'].apply(
    #     lambda x: 1 if x == " Sbrkr" else 0)
    # dataframe['BsmtFullBath'] = dataframe['BsmtFullBath'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # dataframe['HalfBath'] = dataframe['HalfBath'].apply(lambda x: 1 if x > 0 else 0)
    # dataframe['Fireplaces'] = dataframe['Fireplaces'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # dataframe['Functional'] = dataframe['Functional'].apply(
    #     lambda x: 1 if x == 'Typ' else 0)
    # dataframe['PoolArea'] = dataframe['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    # dataframe['WoodDeckSF'] = dataframe['WoodDeckSF'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # dataframe['ScreenPorch'] = dataframe['ScreenPorch'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # dataframe['OpenPorchSF'] = dataframe['OpenPorchSF'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # dataframe['EnclosedPorch'] = dataframe['EnclosedPorch'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # dataframe['3SsnPorch'] = dataframe['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)

    # # New Features
    # dataframe['Years_Since_Remodel'] = dataframe['YrSold'].astype(
    #     int) - dataframe['YearRemodAdd'].astype(int)
    # dataframe['Total_Home_Quality'] = dataframe['OverallQual'] * dataframe['OverallCond']
    # dataframe['Total_Exterior_Quality'] = dataframe['ExterQual'] * dataframe['ExterCond']
    # dataframe['Total_Kitchen_Quality'] = dataframe['KitchenQual'] * dataframe['KitchenAbvGr']
    # dataframe['Total_Fireplace_Quality'] = dataframe['FireplaceQu'] * dataframe['Fireplaces']
    # dataframe['Total_Garage_Quality'] = dataframe['GarageQual'] * dataframe['GarageArea']
    # dataframe['Total_Pool_Quality'] = dataframe['PoolArea'] * dataframe['PoolQC']
    # dataframe['Total_ Bath'] = dataframe['BsmtFullBath'] + \
    #     (.5 * dataframe['BsmtHalfBath']) + \
    #     dataframe['FullBath'] + (.5 * dataframe['HalfBath'])
    # dataframe['Total_Porch_SF'] = dataframe['OpenPorchSF'] + \
    #     dataframe["EnclosedPorch"] + dataframe["3SsnPorch"] + dataframe["ScreenPorch"]
    # dataframe['Total_SF'] = dataframe["GrLivArea"] + dataframe["TotalBsmtSF"]
    # dataframe["Total_FL_SF"] = dataframe["1stFlrSF"] + dataframe["2ndFlrSF"]
    # dataframe['Total_Interior_SF'] = dataframe['TotalBsmtSF'] + \
    #     dataframe['1stFlrSF'] + dataframe['2ndFlrSF']

    # dataframe["Simple_Home_Quality"] = dataframe["Simplified_OverallQual"] * \
    #     dataframe["Simplified_OverallCond"]
    # dataframe["Simple_Exterior_Quality"] = dataframe["Simplified_ExterQual"] * \
    #     dataframe["Simplified_ExterCond"]
    # dataframe["Simple_Pool_Quality"] = dataframe["PoolArea"] * \
    #     dataframe["Simplified_PoolQC"]
    # dataframe["Simple_Garage_Quality"] = dataframe["GarageArea"] * \
    #     dataframe["Simplified_GarageQual"]
    # dataframe["Simple_Fireplace_Quality"] = dataframe["Fireplaces"] * \
    #     dataframe["Simplified_FireplaceQu"]
    # dataframe["Simple_Kitchen_Quality"] = dataframe["KitchenAbvGr"] * \
    #     dataframe["Simplified_KitchenQual"]


    return dataframe