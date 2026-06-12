#:copy of file Decomposer.py:
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
from fugue import transform
import matplotlib.pyplot as plt

class Decomposer:

    """
    Decomposer class
    compute different metrics and decompositions of a time series or catalogs of time series. 
    plot funcitons for individual series analysis are also provided
    its methods can be used as a static class or as an instance to call atributes to reduce recalculation of metrics,
    the catalog calcultaions are static and can run with static or instance strategy
    """

    def __init__(self, t_series: pd.Series):
       
        """
        Constructor for Decomposer instance and avoid recalculation of metrics
        pre: assumes numeric types only
        """
        self.t_series = t_series #TODO support numeric arraylikes o numpy arrays 
        self.adfuller = self.__adfuller__()
        self.adf_statistic = self.__adf_statistic__()
        self.adf_p_value = self.__adf_p_value__()
        self.stationarity = self.__stationarity__()
        self.is_stationary = self.__is_stationary__()
        self.is_non_stationary = self.__is_non_stationary__()
        self.metrics = [self.adf_statistic,
                        self.adf_p_value,
                        self.stationarity, 
                        self.is_stationary, 
                        self.is_non_stationary]

    def __adfuller__(self) -> list:
        try:
            return adfuller(self.t_series)
        except Exception as e: # convergence error or too short series
            #print(e)
            return [10001.0, 10001.0]
    
    def __adf_statistic__(self) -> float:
        return self.adfuller[0]
    
    def __adf_p_value__(self) -> float:
        return  self.adfuller[1]
    
    def __stationarity__(self) -> str:
        if  self.adfuller[1] == 10001.0:
            return "undefined"
        elif  self.adfuller[1] < 0.05:
            return "stationary"
        else:
            return "non-stationary"
        
    def __is_stationary__(self) -> int:
        return 1 if self.stationarity == 'stationary' else 0
    
    def __is_non_stationary__(self) -> int:
        return 1 if self.stationarity == 'non-stationary' else 0
    
    def __series_compute_metrics__(self) -> list: 
        return self.metrics

     
    # STATIC METRICS COMPUTATION
    @staticmethod
    def adf_statistic(t_series: pd.Series) -> float:
        try:
            return adfuller(t_series)[0]
        except Exception as e: # convergence error or too short series
            return 10001.0

    @staticmethod
    def adf_p_value( t_series: pd.Series) -> float:
        try:
            return adfuller(t_series)[1]
        except Exception as e: # convergence error or too short series
            return 10001.0
    
    @staticmethod
    def stationarity(t_series: pd.Series) -> str:
        if Decomposer.adf_p_value(t_series) == 10001.0:
            return "undefined"    
        elif Decomposer.adf_p_value(t_series) < 0.05:
            return "stationary"
        else:
            return "non-stationary"
    
    @staticmethod
    def is_stationary(t_series: pd.Series) -> int:
        return 1 if Decomposer.stationarity(t_series) == 'stationary' else 0

    @staticmethod
    def is_non_stationary(t_series: pd.Series) -> int:
        return 1 if Decomposer.stationarity(t_series) == 'non-stationary' else 0
     
    metrics = [adf_statistic,
              adf_p_value, 
              stationarity, 
              is_stationary, 
              is_non_stationary]
    
    metrics_names = [method.__name__ for method in metrics] 

    @staticmethod
    def series_compute_metrics(t_series: pd.Series) -> list:
        return [f(t_series) for f in Decomposer.metrics]
    
    # @staticmethod
    # def __static_catalog_compute_metrics__( df: pd.DataFrame, ts_id_col: str, target_col:str, date_col:str) -> pd.DataFrame:
    #     df.sort_values(by=[ts_id_col, date_col], ascending=[False, True], inplace=True)
    #     return df.groupby(ts_id_col)[target_col].agg(Decomposer.metrics).reset_index()
    
    @staticmethod
    def __instance_catalog_compute_metrics__(df: pd.DataFrame, ts_id_col: str, target_col:str, date_col:str) -> pd.DataFrame:
        series = df.groupby(ts_id_col)[target_col].agg(lambda ts: Decomposer(ts).metrics)
        return pd.DataFrame(series.tolist(), columns=Decomposer.metrics_names, index=series.index).reset_index()

    @staticmethod
    def catalog_compute_metrics(p_ts_id_col: str, p_target_col:str, p_date_col:str, catalog: pd.DataFrame, runtime_engine:str) -> pd.DataFrame:
        catalog.reset_index(drop=True, inplace=True)
        result = transform(
            df=catalog,
            partition={"by": p_ts_id_col , "presort": f"{p_date_col} asc"},
            using=Decomposer.__instance_catalog_compute_metrics__,
            schema=f"{p_ts_id_col}:str, adf_statistic:double, adf_p_value:double, stationarity:str, is_stationary:int, is_non_stationary:int",
            params=dict(ts_id_col=p_ts_id_col, target_col=p_target_col, date_col=p_date_col),
            engine=runtime_engine,
            as_local=True #TODO debuging
        )
        return result
    
    #most natural periods for every frequency - like defaults
    freq_to_period = {  's': 60, #minutely seasonality
                        'm': 60, #hourly seasonality
                        'H': 24, #daily seasonality
                        'D': 7, #weekly seasonality
                        'W': 52, #yearly seasonality
                        'M': 12, #yearly seasonality
                        'Q': 4, #yearly seasonality
                        'Y': 1} 

    #equivalences between frequencies
    freq_equivalences = {   's': {
                                'm': 60,
                                'H': 3600
                            },
                            'm': {
                                'H': 60,
                                'D': 1440,
                                'W': 10080
                            },
                            'H': {
                                'D': 24,
                                'W': 168
                            },
                            'D': {
                                'W': 7,
                                'M': 30,
                                'Q': 90,
                                'Y': 365
                            },
                            'W': {
                                'M': 4,
                                'Q': 13,
                                'Y': 52
                            },
                            'M': {
                                'Q': 3,
                                'Y': 12
                            },
                            'Q': {
                                'Y': 4
                            }
    }
    
    #equivalences for seasonality computation
    freq_to_seasonals = {   #must be odd numbers to avoid seasonal  error in STL decomposition
                            'H': {
                                'D': 25,
                                'W': 169 # Not likely to be used ?
                            },
                            'D': {
                                "W": 7,
                                'M': 31, 
                                "Y": 365 # Not likely to be used ?
                            },
                            'W': {
                                'M': 7, #actually 4 but 7 is minimun
                                'Q': 13,
                                'Y': 53
                            },
                            'M': {
                                'Q': 7, #actually 3 but 7 is minimun
                                'Y': 13
                            },
                            'Q': {
                                'Y': 7 #actally 4 but 7 is minimun                              
                            }                        
    }
                         
    # CLASSIC DECOMPOSITION

    @staticmethod
    def series_classic_decomposition(t_series: pd.Series, data_freq:str, seasonal_freq:str, p_model='additive') -> pd.DataFrame:
        
        sufix = f"dfreq_{data_freq}_sfreq_{seasonal_freq}_model_{p_model}"
        pPeriod = Decomposer.freq_equivalences[data_freq][seasonal_freq]
        columns_keys = [f'd_classic_trend_{sufix}',
                        f'd_classic_seasonal_{sufix}', 
                        f'd_classic_residual_{sufix}']
        try:
            decompositions = seasonal_decompose(x=t_series, period=pPeriod, model=p_model, extrapolate_trend='freq')
            decompositions = [decompositions.trend, decompositions.seasonal, decompositions.resid]
            return pd.concat(decompositions,
                            axis=1, 
                            keys=columns_keys)
        except Exception as e:
            print(e)
            return pd.DataFrame(columns=columns_keys)
        
    @staticmethod
    def __static_catalog_classic_decomposition__(df: pd.DataFrame, ts_id_col: str, date_col: str, target_col:str, data_freq: str, seasonal_freq:str, p_model='additive') -> pd.DataFrame:
        results = []
        #df.sort_values(by=[ts_id_col, date_col], ascending=[False, True], inplace=True)
        for ts_id, group in df.groupby(ts_id_col):
            decompositions = Decomposer.series_classic_decomposition(group[target_col], data_freq, seasonal_freq, p_model)
            decompositions[ts_id_col] = group[ts_id_col]
            decompositions[date_col] = group[date_col]
            results.append(decompositions)
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def catalog_classic_decomposition( p_df: pd.DataFrame, p_ts_id_col : str, p_date_col :str, p_target_col:str,  p_data_freq: str, p_seasonal_freq, runtime_engine:str, p_pModel='additive') -> pd.DataFrame:
        sufix = f"dfreq_{p_data_freq}_sfreq_{p_seasonal_freq}_model_{p_pModel}: double"
        result = transform(
            df=p_df[[p_ts_id_col, p_date_col, p_target_col]],
            partition={"by": p_ts_id_col , "presort": f"{p_date_col} asc"},
            using=Decomposer.__static_catalog_classic_decomposition__,
            schema= (
                f"{p_ts_id_col}: str,"
                f"{p_date_col}: datetime," 
                f"d_classic_trend_{sufix},"
                f"d_classic_seasonal_{sufix},"
                f"d_classic_residual_{sufix}"
            ),
            params=dict(ts_id_col=p_ts_id_col,
                        date_col=p_date_col,
                        target_col=p_target_col,
                        data_freq=p_data_freq,
                        seasonal_freq=p_seasonal_freq,
                        p_model=p_pModel),
            engine=runtime_engine,
            as_local=True #TODO debuging to print pandas dataframe
        )
        return result
       
        
    @staticmethod
    def classic_decomposition_plot(t_series: pd.Series, data_freq: str, seasonal_freq:str, pModel='additive'):  
        pPeriod = Decomposer.freq_equivalences[data_freq][seasonal_freq]
        try:
            decomposer = seasonal_decompose(x=t_series, period=pPeriod, model=pModel, extrapolate_trend='freq')
            decomposer.plot()
            return decomposer
            
        except Exception as e:
            print(e)
            return None
        
    
    # STL DECOMPOSITION

    @staticmethod
    def series_stl_decomposition(t_series: pd.Series, data_freq: str, seasonal_freq:str, pRobust:bool =True, degree:int = 0, optimized:bool=True ) -> pd.DataFrame:

        sufix = f"dfreq_{data_freq}_sfreq_{seasonal_freq}_robust_{pRobust}_degree_{degree}_optimized_{optimized}"
        p_period = Decomposer.freq_equivalences[data_freq][seasonal_freq]
        p_seasonal = Decomposer.freq_to_seasonals[data_freq][seasonal_freq]
        columns_keys = [f'd_stl_trend_{sufix}',
                        f'd_stl_seasonal_{sufix}', 
                        f'd_stl_residual_{sufix}']
        try:
            p_low_pass_jump = p_seasonal_jump = p_trend_jump = 1
            if optimized:
                p_low_pass_jump = p_seasonal_jump = int(0.15 * (p_period + 1))
                p_trend_jump = int(0.15 * 1.5 * (p_period + 1))

            decompositions = STL(t_series, period=p_period, seasonal=p_seasonal, robust=pRobust, seasonal_deg=degree, trend_deg=degree, low_pass_deg=degree, low_pass_jump=p_low_pass_jump, seasonal_jump=p_seasonal_jump, trend_jump=p_trend_jump).fit()        
            decompositions = [decompositions.trend, decompositions.seasonal, decompositions.resid]
            return pd.concat(decompositions,
                            axis=1, 
                            keys=columns_keys)

        except:
            return pd.DataFrame(columns=columns_keys)
        
    @staticmethod
    def __static_catalog_stl_decomposition__(df: pd.DataFrame, ts_id_col: str, date_col: str, target_col:str, data_freq: str, seasonal_freq:str, pRobust:bool =True, degree:int = 0, optimized:bool=True) -> pd.DataFrame:
        results = []
        #df.sort_values(by=[ts_id_col, date_col], ascending=[False, True], inplace=True)  
        for ts_id, group in df.groupby(ts_id_col):
            decompositions = Decomposer.series_stl_decomposition(group[target_col], data_freq, seasonal_freq, pRobust, degree, optimized)
            decompositions[ts_id_col] = group[ts_id_col]
            decompositions[date_col] = group[date_col]
            results.append(decompositions)
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def catalog_stl_decomposition( p_df: pd.DataFrame, p_ts_id_col : str, p_date_col :str, p_target_col:str,  p_data_freq: str, p_seasonal_freq:str, runtime_engine:str, p_pRobust:bool =True, p_degree:int = 0, p_optimized:bool=True) -> pd.DataFrame:
        sufix = f"dfreq_{p_data_freq}_sfreq_{p_seasonal_freq}_robust_{p_pRobust}_degree_{p_degree}_optimized_{p_optimized}: double"
        result = transform(
            df=p_df[[p_ts_id_col, p_date_col, p_target_col]],
            partition={"by": p_ts_id_col , "presort": f"{p_date_col} asc"},
            using=Decomposer.__static_catalog_stl_decomposition__,
            schema= (
                f"{p_ts_id_col}: str,"
                f"{p_date_col}: datetime," 
                f"d_stl_trend_{sufix},"
                f"d_stl_seasonal_{sufix},"
                f"d_stl_residual_{sufix}"
            ),
            params=dict(ts_id_col=p_ts_id_col,
                        date_col=p_date_col,
                        target_col=p_target_col,
                        data_freq=p_data_freq,
                        seasonal_freq=p_seasonal_freq,
                        pRobust=p_pRobust,
                        degree=p_degree,
                        optimized=p_optimized),
            engine=runtime_engine,
            as_local=True #TODO debuging
        )
        return result
         
    @staticmethod
    def stl_decomposition_plot(t_series: pd.Series, data_freq: str, seasonal_freq:str, pRobust:bool =True, degree:int = 0, optimized:bool=True ):
        try:
            p_period = Decomposer.freq_equivalences[data_freq][seasonal_freq]
            p_seasonal = Decomposer.freq_to_seasonals[data_freq][seasonal_freq]
            low_pass_jump = seasonal_jump = trend_jump = 1
            if optimized:
                low_pass_jump = seasonal_jump = int(0.15 * (p_period + 1))
                trend_jump = int(0.15 * 1.5 * (p_period + 1))
            decomposer = STL(t_series, period=p_period, seasonal=p_seasonal, robust=pRobust, seasonal_deg=degree, trend_deg=degree, low_pass_deg=degree, low_pass_jump=low_pass_jump, seasonal_jump=seasonal_jump, trend_jump=trend_jump).fit()
            decomposer.plot()
        
            return decomposer
        except:
            return None
        
    
    #MSTL DECOMPOSITION

    @staticmethod
    def series_mstl_decomposition(tseries: pd.Series, data_freq: str, seasonal_freqs:list, pRobust:bool =True, degree:int = 0, optimized:bool=True ) -> pd.DataFrame:

        periods = [Decomposer.freq_equivalences[data_freq][seasonal_freq] for seasonal_freq in seasonal_freqs]
        windows = [Decomposer.freq_to_seasonals[data_freq][seasonal_freq] for seasonal_freq in seasonal_freqs]

        p_low_pass_jump = p_seasonal_jump = p_trend_jump = 1
        if optimized:
            #TODO review average aproach for multiple periods
            p_low_pass_jump = p_seasonal_jump = int(sum([0.15 * (period + 1) for period in periods])/len(periods) )                         
            p_trend_jump = int(sum([0.15 * 1.5 * (period + 1) for period in periods])/len(periods) )                    
                        
          
        stl_args = {
            'robust': pRobust,
            'seasonal_deg': degree,
            'trend_deg':degree,
            'low_pass_deg':degree,
            'low_pass_jump':p_low_pass_jump,
            'seasonal_jump':p_seasonal_jump,
            'trend_jump':p_trend_jump
        }

        seasonal_sufix = ''.join(seasonal_freqs)# ray schema does not support '-'

        sufix = f"dfreq_{data_freq}_sfreqs_{seasonal_sufix}_robust_{pRobust}_degree_{degree}_optimized_{optimized}"
        seasonal_name_list = ([f"d_mstl_seasonal_{Decomposer.freq_equivalences[data_freq][s_freq]}_{sufix}"
                                for s_freq
                                in seasonal_freqs])

        seasonal_name_convert = ({f"seasonal_{Decomposer.freq_equivalences[data_freq][s_freq]}": f"d_mstl_seasonal_{Decomposer.freq_equivalences[data_freq][s_freq]}_{sufix}"
                                   for s_freq 
                                   in seasonal_freqs})

        columns_keys = [f'd_mstl_trend_{sufix}',
                        f'd_mstl_residual_{sufix}']

        try:
            decomposer = MSTL(tseries, periods=periods, windows=windows, stl_kwargs=stl_args).fit()
            seasonals = decomposer.seasonal
            df_seasonals = pd.DataFrame()

            if isinstance(seasonals, pd.Series):
              
                # Convert Series to DataFrame
                df_seasonals = seasonals.to_frame()

                # Add missing column with null values
                for col in [f"seasonal_{Decomposer.freq_equivalences[data_freq][s_freq]}" for s_freq in seasonal_freqs]:
                    if col not in df_seasonals.columns:
                        df_seasonals[col] = None
                #rename seasonal column to seasonal_default
                df_seasonals.rename(columns={"seasonal": f"d_mstl_seasonal_default_{sufix}"}, inplace=True)
                
            elif isinstance(seasonals, pd.DataFrame):
                df_seasonals = seasonals
                #add deafult seasonal column
                df_seasonals[f"d_mstl_seasonal_default_{sufix}"] = None

            df_seasonals.rename(columns=seasonal_name_convert, inplace=True)
            decompositions = [decomposer.trend, decomposer.resid]
            decompositions = pd.concat(decompositions,
                            axis=1, 
                            keys=columns_keys)
            return pd.concat([decompositions, df_seasonals], axis=1)
        except Exception as e:
            #print(e)
            return pd.DataFrame(columns=(columns_keys + 
                                         seasonal_name_list +
                                         [f"d_mstl_seasonal_default_{sufix}"]))

        
    @staticmethod
    def __static_catalog_mstl_decomposition__(df: pd.DataFrame, ts_id_col: str, date_col: str, target_col:str, data_freq: str, seasonal_freqs:list, pRobust:bool =True, degree:int = 0, optimized:bool=True) -> pd.DataFrame:
        results = []
        for ts_id, group in df.groupby(ts_id_col):
            decompositions = Decomposer.series_mstl_decomposition(group[target_col], data_freq, seasonal_freqs, pRobust, degree, optimized)
            decompositions[ts_id_col] = group[ts_id_col]
            decompositions[date_col] = group[date_col]
            results.append(decompositions)
        
        return pd.concat(results, ignore_index=True)  
    
    @staticmethod
    def catalog_mstl_decomposition( p_df: pd.DataFrame, p_ts_id_col : str, p_date_col :str, p_target_col:str,  p_data_freq: str, p_seasonal_freqs:list, runtime_engine:str, p_pRobust:bool =True, p_degree:int = 0, p_optimized:bool=True) -> pd.DataFrame:
        sufix = f"dfreq_{p_data_freq}_sfreqs_{''.join(p_seasonal_freqs)}_robust_{p_pRobust}_degree_{p_degree}_optimized_{p_optimized}: double"
        seasonal_name_list = ([f"d_mstl_seasonal_{Decomposer.freq_equivalences[p_data_freq][s_freq]}_{sufix}"
                                for s_freq
                                in p_seasonal_freqs])
        seasonal_name_list = ','.join(seasonal_name_list)

        result = transform(
            df=p_df,
            partition={"by": p_ts_id_col, "presort": f"{p_date_col} asc"},
            using=Decomposer.__static_catalog_mstl_decomposition__,
            schema= (
                f"{p_ts_id_col}: str,"
                f"{p_date_col}: datetime,"
                f"d_mstl_trend_{sufix},"
                f"d_mstl_residual_{sufix},"
                f"{seasonal_name_list},"
                f"d_mstl_seasonal_default_{sufix}"
            ),
            params=dict(ts_id_col=p_ts_id_col,
                        date_col=p_date_col,
                        target_col=p_target_col,
                        data_freq=p_data_freq,
                        seasonal_freqs=p_seasonal_freqs,
                        pRobust=p_pRobust,
                        degree=p_degree,
                        optimized=p_optimized),
            engine=runtime_engine,
            as_local=True #TODO debuging
        )
        return result
    
    @staticmethod
    def mstl_decomposition_plot(t_series: pd.Series, data_freq: str, seasonal_freqs:list, pRobust:bool =True, degree:int = 0, optimized:bool=True ):
        try:
            periods = [Decomposer.freq_equivalences[data_freq][seasonal_freq] for seasonal_freq in seasonal_freqs]
            windows = [Decomposer.freq_to_seasonals[data_freq][seasonal_freq] for seasonal_freq in seasonal_freqs]
            p_low_pass_jump = p_seasonal_jump = p_trend_jump = 1
            if optimized:
                p_low_pass_jump = p_seasonal_jump = int(sum([0.15 * (period + 1) for period in periods])/len(periods) )                         
                p_trend_jump = int(sum([0.15 * 1.5 * (period + 1) for period in periods])/len(periods) )                    
            decomposer = MSTL(t_series, periods=periods, windows=windows, stl_kwargs={'robust': pRobust, 'seasonal_deg': degree, 'trend_deg':degree, 'low_pass_deg':degree, 'low_pass_jump':p_low_pass_jump, 'seasonal_jump':p_seasonal_jump, 'trend_jump':p_trend_jump}).fit()
            
            decomposer.plot()
        
            return decomposer    
             
        except:
            return None
