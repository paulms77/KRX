import pandas as pd
import kquant as kq

def fetch_data(args: (str, str, str))->list[pd.DataFrame()]:
    """
    병렬처리를 통한 주식 종목의 일간정보 가져오기

    :param (str, str, str) args: 종목코드, 첫 매매일 날짜, 마지막 매매일 날짜 종보가 담긴 튜플
    :return: start_date~end_date 기간의 symbol_id 종목의 데이터
    :rtype: list[pd.DataFrame()]
    """
    symbol_id, start_date, end_date = args
    
    try:
        data = kq.daily_stock(symbol_id, start_date, end_date)
            
        if data is not None and not data.empty:
            return data
    except Exception as e:
        print(f"Error fetching data for symbol {symbol_id}: {str(e)}")
        return None