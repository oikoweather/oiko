import requests
import json
import pandas as pd
import math
from datetime import timedelta, datetime
import numpy as np

class Oiko():
    def __init__(self, api_key):
        self.api_key = api_key


    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


    def solar_angle(lat, lon, localtime, timezone, day_of_year):
        '''
        Calculate solar angle
        Source: https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF, returns in rad
        '''
        hour = localtime.hour
        minute = localtime.minute

        # Fractional year in radians
        frac_year = 2 * math.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)

        # Equation of time in minutes
        eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(frac_year) -
                           0.032077 * math.sin(frac_year) - 0.014615 * math.cos(2 * frac_year) -
                           0.040849 * math.sin(2 * frac_year))

        # Declination angle in radians
        decl = 0.006918 - 0.399912 * math.cos(frac_year) + 0.070257 * math.sin(frac_year) - 0.006758 * math.cos(2 * frac_year) + 0.000907 * math.sin(
            2 * frac_year) - 0.002697 * math.cos(3 * frac_year) + 0.00148 * math.sin(3 * frac_year)

        # Time offset in minutes
        time_offset = eqtime + 4 * lon - 60 * timezone

        tst = hour * 60 + minute + time_offset

        # Solar hour angle in degrees
        ha = (tst / 4) - 180

        elev_angle_rad = math.asin(math.sin(math.pi / 180 * lat) * math.sin(decl) + math.cos(math.pi / 180 * lat) *
                                   math.cos(decl) * math.cos(math.pi / 180 * ha))

        return max(elev_angle_rad, 0)


    def get_illuminance(ghi, dhi, dni, zenith, rel_airmass, dewPoint):
        '''
        Source: Generate illuminance values per 'Modeling Daylight Availability and Irradiance Components from
        Direct and Global Irradiance', Perez R. (1990)

        glbeIll, drctIll, diffIll, zenIll = get_illuminance(ghi,dhi,dni,zenith_deg,rel_airmass,dewPoint)

        Note: Currently uses actual zenith rather than apparent zenith
        '''

        if zenith < math.pi / 2 - (0.5 * math.pi / 180.) and ghi > 0:
            kai = 1.041
            eps = ((dhi + dni) / dhi + kai * zenith ** 3) / (1 + kai * zenith ** 3)
            delta = dhi * rel_airmass / 1360
            W = math.exp(0.08 * dewPoint - 0.075)

            # Per Perez Table 1: Discrete Sky Clearness Categories
            if eps >= 1 and eps < 1.065:
                e_category = 1
            elif eps >= 1.065 and eps < 1.23:
                e_category = 2
            elif eps >= 1.23 and eps < 1.5:
                e_category = 3
            elif eps >= 1.5 and eps < 1.95:
                e_category = 4
            elif eps >= 1.95 and eps < 2.8:
                e_category = 5
            elif eps >= 2.8 and eps < 4.5:
                e_category = 6
            elif eps >= 4.5 and eps < 6.2:
                e_category = 7
            elif eps >= 6:
                e_category = 8
            else:
                print('error')
                return None

            # Per Perez Table 4: Luminous Efficacy
            GlobLumEff = {}
            GlobLumEff[1] = (96.63, -0.47, 11.50, -9.16)
            GlobLumEff[2] = (107.54, 0.79, 1.79, -1.19)
            GlobLumEff[3] = (98.73, 0.70, 4.40, -6.95)
            GlobLumEff[4] = (92.72, 0.56, 8.36, -8.31)
            GlobLumEff[5] = (86.73, 0.98, 7.10, -10.94)
            GlobLumEff[6] = (88.34, 1.39, 6.06, -7.60)
            GlobLumEff[7] = (78.63, 1.47, 4.93, -11.37)
            GlobLumEff[8] = (99.65, 1.86, -4.46, -3.15)

            # Eq 6
            a, b, c, d = GlobLumEff[e_category]
            glbeIll = int(ghi * (a + b * W + c * math.cos(zenith) + d * math.log(delta)))

            DirLumEff = {}
            DirLumEff[1] = (57.20, -4.55, -2.98, 117.12)
            DirLumEff[2] = (98.99, -3.46, -1.21, 12.38)
            DirLumEff[3] = (109.83, -4.90, -1.71, -8.81)
            DirLumEff[4] = (110.34, -5.84, -1.99, -4.56)
            DirLumEff[5] = (106.36, -3.97, -1.75, -6.16)
            DirLumEff[6] = (107.19, -1.25, -1.51, -26.73)
            DirLumEff[7] = (105.75, 0.77, -1.26, -34.44)
            DirLumEff[8] = (101.18, 1.58, -1.10, -8.29)

            # Eq 8
            a, b, c, d = DirLumEff[e_category]
            drctIll = int(max(0, dni * (a + b * W + c * math.exp(5.73 * zenith - 5) + d * delta)))

            DiffLumEff = {}
            DiffLumEff[1] = (97.24, -0.46, 12.00, -8.91)
            DiffLumEff[2] = (107.22, 1.15, 0.59, -3.95)
            DiffLumEff[3] = (104.97, 2.96, -5.52, -8.77)
            DiffLumEff[4] = (102.39, 5.59, -13.95, -13.90)
            DiffLumEff[5] = (100.71, 5.94, -22.75, -23.74)
            DiffLumEff[6] = (106.42, 3.83, -36.15, -28.83)
            DiffLumEff[7] = (141.88, 1.90, -53.24, -14.03)
            DiffLumEff[8] = (152.23, 0.35, -45.27, -7.98)

            # Eq 7
            a, b, c, d = DiffLumEff[e_category]
            diffIll = int(dhi * (a + b * W + c * math.cos(zenith) + d * math.log(delta)))

            ZenLumEff = {}
            ZenLumEff[1] = (40.86, 26.77, -29.59, -45.75)
            ZenLumEff[2] = (26.58, 14.73, 58.46, -21.25)
            ZenLumEff[3] = (19.34, 2.28, 100.00, 0.25)
            ZenLumEff[4] = (13.25, -1.39, 124.79, 15.66)
            ZenLumEff[5] = (14.47, -5.09, 160.09, 9.13)
            ZenLumEff[6] = (19.76, -3.88, 154.61, -19.21)
            ZenLumEff[7] = (28.39, -9.67, 151.58, -69.39)
            ZenLumEff[8] = (42.91, -19.62, 130.80, -164.08)

            # Eq 9
            a, b, c, d = ZenLumEff[e_category]
            zenIll = int(dhi * (a + b * math.cos(zenith) + c * math.exp(-3 * zenith) + d * delta))

        # If sun is below horizon, return 0
        else:
            glbeIll, drctIll, diffIll, zenIll = 0, 0, 0, 0

        return glbeIll, drctIll, diffIll, zenIll


    def get_epw_data(self, lat, lon, year, city, state, country, filename):
        '''
        Get historical weather parameters to create EPW file for the specified location
        '''

        parameters = ['temperature', 'dewpoint_temperature', 'surface_solar_radiation', 'surface_thermal_radiation',
                      'surface_direct_solar_radiation', 'surface_diffuse_solar_radiation', 'direct_normal_solar_radiation',
                      'relative_humidity', 'wind_speed', 'wind_direction', 'surface_pressure', 'total_cloud_cover', 'total_precipitation',
                      'soil_temperature_level_3', 'soil_temperature_level_4', 'forecast_albedo', 'cloud_base_height', 'total_column_rain_water',
                      'snow_depth', 'snow_density', 'snowfall']

        start = f'{year - 1}-12-31'
        end = f'{year + 1}-01-02'

        if is_leap_year(year):
            leap_year = 'Yes'
        else:
            leap_year = 'No'

        r = requests.get('https://api.oikolab.com/weather', params={'param': parameters, 'lat': lat, 'lon': lon, 'start': start, 'end': end},
                         headers={'content-encoding': 'gzip', 'Connection': 'close', 'api-key': self.api_key})

        weather_data = json.loads(r.json()['data'])
        epw_df = pd.DataFrame(index=pd.to_datetime(weather_data['index'], unit='s'),
                              data=weather_data['data'],
                              columns=weather_data['columns'])
        epw_df.index.name = 'datetime'
        utc_offset = epw_df['utc_offset (hrs)'][0]

        epw_df.index = epw_df.index + timedelta(hours=utc_offset)
        epw_df = epw_df[epw_df.index.year == year]

        epw_columns = ['year', 'month', 'day', 'hour', 'min', 'flag', 'temperature', 'dewPoint', 'RH', 'pressure_Pa',
                       'extGHI', 'extDNI', 'DWR', 'ghi', 'dni', 'dhi', 'glbeIll', 'drctIll', 'diffIll', 'zenIll',
                       'windBearing', 'windSpeed', 'totSkyCover', 'opqSkyCover', 'visibility', 'ceiling',
                       'Obs', 'weatherCode', 'precH2O', 'OpticalDepth', 'snowDepth', 'DaysSinceSnow', 'albedo', 'liqDepth', 'liqQty']

        epw_df['year'] = epw_df.index.year
        epw_df['month'] = epw_df.index.month
        epw_df['day'] = epw_df.index.day
        epw_df['hour'] = epw_df.index.hour + 1
        epw_df['min'] = epw_df.index.minute
        epw_df['flag'] = '?9?9?9?9E0?9?9?9?9*9?9?9?9?9?9?9?9?9?9*_*9*9*9?9?9'
        epw_df['ceiling'] = epw_df['cloud_base_height (m)'].fillna(value='77777')
        epw_df['ceiling'] = epw_df['ceiling'].astype(int)

        # Fill missing values if required (rare case but needed for integer conversion)
        epw_df = epw_df.interpolate(limit_direction='both', limit=6)

        epw_df['temperature'] = epw_df['temperature (degC)']
        epw_df['dewPoint'] = epw_df['dewpoint_temperature (degC)']
        epw_df['RH'] = (epw_df['relative_humidity (0-1)'] * 100).astype(int)
        epw_df['pressure_Pa'] = (epw_df['surface_pressure (Pa)']).astype(int)
        epw_df['extGHI'] = 9999
        epw_df['extDNI'] = 9999
        epw_df['DWR'] = epw_df['surface_thermal_radiation (W/m^2)'].astype(int)
        epw_df['ghi'] = epw_df['surface_solar_radiation (W/m^2)'].astype(int)
        epw_df['dni'] = epw_df['direct_normal_solar_radiation (W/m^2)'].astype(int)
        epw_df['dhi'] = epw_df['surface_diffuse_solar_radiation (W/m^2)'].astype(int)
        epw_df['windBearing'] = epw_df['wind_direction (deg)'].astype(int)
        epw_df['windSpeed'] = epw_df['wind_speed (m/s)']
        epw_df['totSkyCover'] = (epw_df['total_cloud_cover (0-1)'] * 10).astype(int)
        epw_df['opqSkyCover'] = (epw_df['total_cloud_cover (0-1)'] * 10).astype(int)
        epw_df['visibility'] = 9999
        epw_df['Obs'] = 0
        epw_df['weatherCode'] = 999999999
        epw_df['precH2O'] = epw_df['total_column_rain_water (mm of water equivalent)']
        epw_df['OpticalDepth'] = 999
        epw_df['snowDepth'] = epw_df['snow_depth (mm of water equivalent)'] / epw_df['snow_density (kg/m^3)'] * 100
        epw_df['DaysSinceSnow'] = 99
        epw_df['albedo'] = epw_df['forecast_albedo (0-1)']
        epw_df['liqDepth'] = (epw_df['total_precipitation (mm of water equivalent)'] - epw_df['snowfall (mm of water equivalent)'])
        epw_df['liqDepth'] = epw_df['liqDepth'].where(epw_df['liqDepth'] > 0, 0)
        epw_df['liqQty'] = 999
        epw_df['dayofyear'] = epw_df.index.dayofyear
        epw_df['lat'] = lat
        epw_df['lon'] = lon
        epw_df['timezone'] = utc_offset

        soil1 = list(epw_df['soil_temperature_level_3 (degC)'].resample('M').mean().round(1))
        soil1 = f'{soil1}'.replace('[', '')
        soil1 = soil1.replace(']', '')
        soil1 = soil1.replace(' ', '')

        soil2 = list(epw_df['soil_temperature_level_4 (degC)'].resample('M').mean().round(1))
        soil2 = f'{soil2}'.replace('[', '')
        soil2 = soil2.replace(']', '')
        soil2 = soil2.replace(' ', '')

        epw_df = epw_df.reset_index()
        epw_df['solar_angle'] = epw_df.apply(lambda x: solar_angle(x.lat, x.lon, x.datetime, x.timezone, x.dayofyear), axis=1)
        epw_df['rel_airmass'] = 1 / np.cos(epw_df['solar_angle'])
        epw_df[['glbeIll', 'drctIll', 'diffIll', 'zenIll']] = epw_df.apply(
            lambda x: get_illuminance(x.ghi, x.dhi, x.dni, x.solar_angle, x.rel_airmass, x.dewPoint),
            axis=1, result_type="expand")

        dayname = epw_df['datetime'][0].day_name()
        elev = epw_df['model elevation (surface)'][0]

        line1 = f'LOCATION,{city},{state},{country},ERA5 (ECMWF),n/a,{lat},{lon},{utc_offset},{elev}\n'
        line2 = f'DESIGN CONDITIONS, 0\n'
        line3 = f'TYPICAL/EXTREME PERIODS, 0\n'
        line4 = f'GROUND TEMPERATURES,2,.5,,,,{soil1},2,,,,{soil2}\n'
        line5 = f'HOLIDAYS/DAYLIGHT SAVINGS,{leap_year},0,0,0\n'
        line6 = f'COMMENTS 1, EPW file generated by Oikolab (https://oikolab.com) with ECMWF ERA5 Reanalysis dataset\n'
        line7 = f'COMMENTS 2, Please contact support@oikolab.com for any questions.\n'
        line8 = f'DATA PERIODS,1,1,Data,{dayname}, 1/ 1,12/31\n'

        with open(filename, 'w') as f:

            f.write(line1)
            f.write(line2)
            f.write(line3)
            f.write(line4)
            f.write(line5)
            f.write(line6)
            f.write(line7)
            f.write(line8)
            epw_df[epw_columns].to_csv(f, header=False, index=False)

        return epw_df

    def get_weather(self,lat=None,lon=None,location='',start=None,end=None,parameters=None,freq='H',resample_method=None,):

        params = {'param': parameters,
                  'lat':lat,'lon':lon,'location':location,'start':start,'end':end,'freq':freq,'resample_method':resample_method, 'format':'csv'}

        r = requests.get('https://api.oikolab.com/weather', params=params,
                         headers={'content-encoding': 'gzip', 'Connection': 'close', 'api-key': self.api_key})

        if r.status_code == 200 or r.status_code==201:
            df = pd.read_csv(r.url,index_col='datetime (UTC)')
            df.index = pd.to_datetime(df.index)
            return df

        else:
            print(r)


