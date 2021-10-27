from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os
import datetime
import time



PATH = "chromedriver.exe"

driver = webdriver.Chrome(PATH)
df = pd.read_csv('equity40.csv')
security_list = df['code'].to_list()

def choose_date(day,month,year):
    year_ = Select(driver.find_element_by_class_name("ui-datepicker-year"))
    year_.select_by_visible_text(year)
    month_ = Select(driver.find_element_by_class_name("ui-datepicker-month"))
    month_.select_by_visible_text(month)
    day_ = driver.find_element_by_link_text(day)
    day_.click()

def download_data(company_name , from_day , from_month , from_year , to_day , to_month, to_year):
    driver.get("https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx")
    search = driver.find_element_by_name("ctl00$ContentPlaceHolder1$smartSearch")
    search.send_keys(company_name)
    search.send_keys(Keys.RETURN)
    time.sleep(1)
    from_input = driver.find_element_by_name("ctl00$ContentPlaceHolder1$txtFromDate")
    from_input.click()
    choose_date(from_day,from_month,from_year)
    to_input = driver.find_element_by_name("ctl00$ContentPlaceHolder1$txtToDate")
    to_input.click()
    choose_date(to_day,to_month,to_year)
    submit_btn = driver.find_element_by_name("ctl00$ContentPlaceHolder1$btnSubmit")
    submit_btn.click()
    download_btn = driver.find_element_by_id("ContentPlaceHolder1_btnDownload1")
    download_btn.click()

def download_monthly_data(company_name , from_month, from_year):

    driver.get("https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx")
    month_active = driver.find_element_by_id("ContentPlaceHolder1_rdbMonthly")
    month_active.click()

    search = driver.find_element_by_name("ctl00$ContentPlaceHolder1$smartSearch")
    search.send_keys(company_name)
    search.send_keys(Keys.RETURN)
    time.sleep(1)
   
    select_month = Select(driver.find_element_by_id("ContentPlaceHolder1_cmbMonthly"))
    select_month.select_by_visible_text(from_month)
    select_year = Select(driver.find_element_by_id("ContentPlaceHolder1_cmbMYear"))
    select_year.select_by_visible_text(from_year)
    submit_btn = driver.find_element_by_name("ctl00$ContentPlaceHolder1$btnSubmit")
    submit_btn.click()
    time.sleep(1)
    download_btn = driver.find_element_by_id("ContentPlaceHolder1_btnDownload1")
    download_btn.click()
    

def rename_files(security , i):
    try:
        os.rename(r"C:/Users/kdivy/Downloads/"+str(security)+".csv", "BSE/daily/" + df['symbol'][i]+".csv")
    except:
        time.sleep(1)
        print("C:/Users/kdivy/Downloads/"+str(security)+".csv")
def rename_monthly_files(security , i):
    try:
        os.rename(r"C:/Users/kdivy/Downloads/"+str(security)+".csv", "BSE/monthly/" + df['symbol'][i]+".csv")
    except:
        time.sleep(1)
        print("C:/Users/kdivy/Downloads/"+str(security)+".csv")

for i in range(0,40):
    #download_data(security_list[i], "1" , "Jan" , "2011" , "1" , "Jan" , "2021")
    download_monthly_data(security_list[i],"Jan" , "2011")
    time.sleep(1)

for i in range(0,40):
    #rename_files(security_list[i] , i)
    rename_monthly_files(security_list[i] , i)