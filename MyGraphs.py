import os
import time
import re


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from collections import *

class MyGraphs:
    def __init__(self, datafolder="",
                 day="",
                 which=None,
                 window=60,
                 cuvette=600,
                 denoise=True,
                 derive=True,
                 norm_chl=True):
        # rawdatafile encoding
        self.encoding = 'ISO-8859-1'

        # variables
        self.datafolder = datafolder
        self.day = day
        self.precision = 8
        self.timecol = 'time'

        # dataframes and dict of dataframes
        self.masterdf = pd.DataFrame()
        self.dfs = {}

        self.style = 'vc'
        self.window = window
        self.cuvette = cuvette

        historycols = ['timestamp', 'event', 'comments']
        self.history = pd.DataFrame(data=np.zeros((0, len(historycols))), columns=historycols)
        self.history.append([time.time(), "start", ""])

        # which graphs to superimpose in compiled graphs ?
        self.which = which

        # files and folders
        self.samples_file = os.path.join(self.datafolder, self.day, self.day + "_SAMPLES.csv")
        self.figures_individual_folder = os.path.join(self.datafolder, self.day, "figures", "individual")
        self.figures_compiled_folder = os.path.join(self.datafolder, self.day, "figures", "compiled")
        self.masterfile = os.path.join(self.datafolder, self.day, "_" + self.day + "_MASTER.csv")
        self.summaryfile = os.path.join(self.datafolder, self.day, "_" + self.day + "_SUMMARY.csv")
        self.samplefile_cleaned = os.path.join(self.datafolder, self.day, "_" + self.day + "_SAMPLE_cleaned.csv")
        self.logfile = os.path.join(self.datafolder, self.day, "_" + self.day + "_log.csv")

        # initialize graphics
        self.dpi = 600  # figure resolution
        self.figformat = 'jpg'  # figure extension
        self.linewidth = 1.2
        self.figWidth = 20
        self.figHeight = 12
        self.annotate = True
        self.legend = True
        self.ymax, self.ymin = None, None
        self.title = True
        self.xlabel = True
        self.ylabel = True

        self.eventColors = {
            "BIC": "#000000",
            "CELLS": "#00aa00",
            "AZ": "#aa0000",
            "EZ": "#0000aa",
            "CUSTOM": "#000000",
            "LIGHT ON": "#eeeea2",
            "LIGHT OFF": "#eeeea2"
        }
        self.curveColors = {
            "Mass32": "#FA0000",
            "Mass32_cons": "#FA0000",
            "Mass40": "#C89600",
            "Mass44": "#00FA00",
            "Mass45": "#32C8FA",
            "Mass46": "#FA96FA",
            "Mass47": "#0064FA",
            "Mass49": "#6400FA",
            "totalCO2": "#000000",
            "logE49": "#646464",
            "d32dt": "#aa0000",
            "d32dt_d": "#ff5555",
            "d32dt_cd": "#ee0000",
             "d32dt_cd_chl": "#ee0000",
            "d32dt_cons": "#FA0000",
            "d40dt": "#C89600",
            "d44dt": "#00FA00",
            "d45dt": "#32C8FA",
            "d46dt": "#FA96FA",
            "d47dt": "#0064FA",
            "d49dt": "#6400FA",
            "d40dt_d": "#C89600",
            "d44dt_d": "#00FA00",
            "d45dt_d": "#32C8FA",
            "d46dt_d": "#FA96FA",
            "d47dt_d": "#0064FA",
            "d49dt_d": "#6400FA",
            "d40dt_chl": "#C89600",
            "d44dt_chl": "#00FA00",
            "d45dt_chl": "#32C8FA",
            "d46dt_chl": "#FA96FA",
            "d47dt_chl": "#0064FA",
            "d49dt_chl": "#6400FA",
            "dtotalCO2dt": "#000000",
            "dtotalCO2dt_d": "#000000",
            "enrichrate47": "#0064FA",
            "enrichrate49": "#6400FA",
            "dtotalCO2dt_chl": "#000000",
            "enrichrate47_chl": "#0064FA",
            "enrichrate49_chl": "#6400FA",
            "cat": "#6400FA"
        }

        self.units = {
            'd40dt': 'V/s',
            'd44dt': 'µmol/L/s',
            'd45dt': 'µmol/L/s',
            'd46dt': 'µmol/L/s',
            'd47dt': 'µmol/L/s',
            'd49dt': 'µmol/L/s',
            'dtotalCO2dt': 'µmol/L/s',
            'd32dt': 'µmol/L/s',
            'd32dt_c': 'µmol/L/s',
            'd32dt_cd': 'µmol/L/s',
            'enrichrate47': 's-1',
            'enrichrate49': 's-1',

            'd40dt_d': 'V/s',
            'd44dt_d': 'µmol/L/s',
            'd45dt_d': 'µmol/L/s',
            'd46dt_d': 'µmol/L/s',
            'd47dt_d': 'µmol/L/s',
            'd49dt_d': 'µmol/L/s',
            'dtotalCO2dt_d': 'µmol/L/s',
            'd32dt_d': 'µmol/L/s',

            'd44dt_chl': 'µmol/s/mg chl',
            'd45dt_chl': 'µmol/s/mg chl',
            'd46dt_chl': 'µmol/s/mg chl',
            'd47dt_chl': 'µmol/s/mg chl',
            'd49dt_chl': 'µmol/s/mg chl',
            'dtotalCO2dt_chl': 'µmol/s/mg chl',
            'd32dt_chl': 'µmol/s/mg chl',
            'd32dt_d_chl': 'µmol/s/mg chl',
            'd32dt_cd_chl': 'µmol/s/mg chl',
            'enrichrate47_chl': '/s/mg chl',
            'enrichrate49_chl': '/s/mg chl',

            'logE47':'',
            'logE49':''
        }

        # Clean samples file
        self.sample_list()

        # Create master file
        self.create_masterdf(derive=derive, denoise=denoise, norm_chl=norm_chl)

    def sample_list(self):
        """
        Cleans the list of samples
        """

        # clean sample file if not already done
        if not os.path.isfile(os.path.join(self.samplefile_cleaned)):
            # get file list
            allsamples = pd.read_csv(self.samples_file, encoding=self.encoding)

            # merge with chlorophyll
            chlorofile = "C:\\Users\\u5040252\\CloudStation\\Projects\\Mass_Spec_Data\\2016\\chlorophylls.csv"
            chl = pd.read_csv(chlorofile, encoding='ISO-8859-1')[['date', 'samplename','chlorophyll']]
            allsamples = allsamples.merge(chl, on=['date', 'samplename'], how='left')


            # fix datetimes
            for i in range(0, len(allsamples)):
                allsamples.loc[i, 'myDateTime'] = pd.to_datetime(
                    str(allsamples.loc[i, 'date']) + " " + allsamples.loc[i, 'time'])
            allsamples.drop(['date', 'time'], axis=1, inplace=True)

            # set file paths
            allsamples.set_index('samplename', inplace=True)
            allsamples['rawdatafile'] = None
            allsamples['datafile'] = None
            allsamples = allsamples.drop_duplicates('filename')


            allsamples['BIC'] = 0
            allsamples['CELLS'] = 0
            allsamples['AZ'] = 0
            allsamples['EZ'] = 0
            allsamples['chloro_ugml'] = 0


            for s in allsamples.index:
                # create file paths
                datafilepath = str(os.path.join(self.datafolder, self.day, str(allsamples.loc[s, 'filename'])))
                rawdatafilepath = str(
                    os.path.join(self.datafolder, self.day, 'rawdata', str(allsamples.loc[s, 'filename'])))

                # check file paths
                allsamples.at[s, 'datafile'] = datafilepath if os.path.isfile(datafilepath) else None
                allsamples.at[s, 'rawdatafile'] = rawdatafilepath if os.path.isfile(rawdatafilepath) else None

                if (allsamples.at[s, 'datafile'] is None) and (allsamples.at[s, 'rawdatafile'] is None):
                    print(s, 'not found')

                # datafile = os.path.join(self.datafolder, self.day, allsamples.at[s, "datafile"])
                if os.path.isfile(datafilepath):
                    data = pd.read_csv(datafilepath, encoding=self.encoding)

                    # parse events
                    for r in data.loc[data.eventtype.notnull(),["time2", "eventtype", "eventdetails"]].iterrows():
                        r = r[1]
                        if r['eventtype'] in ["BIC", "CELLS", "AZ", "EZ"]:
                #             print(s, r['eventtype'], re.search("[0-9]+", r["eventdetails"]).group())
                            allsamples.at[s, r['eventtype']] = float(re.search("[0-9]+", r["eventdetails"]).group())


            # calculate final chlorophyll content
            #     print(s, allsamples.at[s,'CELLS'],allsamples.at[s,'cuvette'],allsamples.at[s,'chlorophyll'], )
            allsamples['chloro_ugml'] = allsamples.CELLS / allsamples.cuvette * allsamples.chlorophyll # µg/mL
            allsamples.chloro_ugml = allsamples.chloro_ugml.apply(lambda x:round(x, 2))




            # remove samples for which we have no datafile
            self.samples = allsamples.ix[allsamples.datafile.notnull(), :]

            # if len(self.samples.index) > 0:
            #     # chlorophyll
            #     if os.path.isfile(os.path.join(self.datafolder, self.day, self.day + "_chlorophyll.csv")):
            #         chldf = pd.read_csv(os.path.join(self.datafolder, self.day, self.day + "_chlorophyll.csv"))
            #         chldf.set_index('sample', inplace=True)
            #         self.samples = self.samples.join(chldf, how='left')
            #     else:
            #         self.samples.loc[:, 'chl'] = 1

            # save cleaned sample file
            self.samples.to_csv(self.samplefile_cleaned, encoding=self.encoding)
            self.log("SAMPLES cleaned", "Created " + self.samplefile_cleaned)

        else:
            self.samples = pd.read_csv(self.samplefile_cleaned, encoding=self.encoding)
            self.samples.set_index('samplename', inplace=True)


    def create_masterdf(self, derive=True, denoise=True, norm_chl=True):
        """
        Create a master dataframe which contains all the data from that day and
        a summary dataframe which contains a list of the events from each experiment

        :param derive: if True, will recalculate all derivatives
        :param derive: if True, will calculate denoised data
        :return:
        """
        if len(self.samples.index) > 0:
            if (not os.path.isfile(self.masterfile)) or derive or denoise or norm_chl:

                masterdf = pd.DataFrame()

                for s in self.samples.index:
                    # if os.path.getsize(self.samples.ix[s, 'datafile']) >= 10000000:
                    #     print("File ", self.samples.ix[s, 'datafile'], " is more than 10 Mo... Excluding it")
                    # else:
                        self.dfs[s] = pd.read_csv(self.samples.ix[s, 'datafile'], encoding=self.encoding)
                        self.dfs[s]['echant'] = s

                        self.timecol = 'time2' if 'time2' in self.dfs[s].columns else 'time'

                        # remove first data point which can cause problems sometimes
                        self.dfs[s] = self.dfs[s].ix[1:, :]
                        self.dfs[s].reset_index(inplace=True, drop=True)

                        if 'O2evol' in self.dfs[s].columns:
                            self.dfs[s] = self.dfs[s].rename(columns={'O2evol': 'd32dt'})

                        # (re)calculate all the derivatives if necessary using time2, self.window and a rolling linear regression
                        if derive:
                            self.dfs[s] = self.derive(self.dfs[s])

                        # denoising
                        if denoise:
                            self.dfs[s] = self.denoise_data(s)

                        if norm_chl and 'CELLS' in set(self.dfs[s].eventtype):
                            self.normalize_chlorophyll(s)
                        # save modified file
                        self.dfs[s].to_csv(self.samples.ix[s, 'datafile'], index=False, encoding=self.encoding)

                        # concat dataframes
                        masterdf = pd.concat([masterdf, self.dfs[s]])

                # save master dataframe in folder
                self.masterdf = masterdf.sort_values(by=['echant', 'time'], ascending=[True, True])
                self.masterdf.to_csv(self.masterfile,  compression='gzip', index=False, encoding=self.encoding)
                self.log("MASTER file", "Created " + self.masterfile)
                masterdf = None

                self.summarydf = self.masterdf.loc[self.masterdf.eventtype.notnull(), :]
                self.summarydf.to_csv(self.summaryfile, index=False, encoding=self.encoding)
                self.log("SUMMARY file", "Created " + self.summaryfile)
                summarydf = None


            else:
                print(self.masterfile, " exists")
                # Load master df
                self.masterdf = pd.read_csv(self.masterfile, compression='gzip', encoding=self.encoding)

                # Load summary df
                self.summarydf = pd.read_csv(self.summaryfile, encoding=self.encoding)

                # load one df per datafile
                for s in self.samples.index:
                    self.dfs[s] = pd.read_csv(self.samples.at[s, 'datafile'], encoding=self.encoding)
                    self.dfs[s]['echant'] = s

    def myround(self, value):
        return round(float(value), self.precision)

    def log(self, event, comment):
        self.history.loc[len(self.history)] = [time.strftime("%Y%m%d - %H:%M:%S"), event, comment]
        self.history.to_csv(self.logfile, encoding=self.encoding, index=False)

    def derive(self, df):
        for m in [32, 40, 44, 45, 46, 47, 49]:
            df.loc[:, 'd' + str(m) + 'dt'] = self.rolling_linear_reg(df[self.timecol], df['Mass' + str(m)])

        df.loc[:, 'dtotalCO2dt'] = df.d44dt + df.d45dt + df.d46dt + df.d47dt + df.d49dt
        df.loc[:, 'enrichrate47'] = self.rolling_linear_reg(df[self.timecol], df.logE47)  # (per second)
        df.loc[:, 'enrichrate49'] = self.rolling_linear_reg(df[self.timecol], df.logE49)  # (per second)

        return df

    def numpy_rolling_window(self, a):
        shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def rolling_linear_reg(self, x, y):
        r = np.zeros(len(x))
        r[:] = np.NAN
        myrange = range(int(self.window / 2), len(x) - int(self.window / 2))

        a = np.vstack([x, np.ones(len(x))]).T
        for i in myrange:
            r[i] = self.myround(np.linalg.lstsq(a[i:i + self.window], y[i:i + self.window])[0][0])
        return r

    def consumption_params(self, s, limit=7200):
        # denoising variables
        self.reffiles = {600: "C:\\Users\\u5040252\\Cloudstation\\Projects\\Mass_Spec_Data\\2016\\20160331\\rawdata\\20160331_cons600_4.csv",
                         2000: "C:\\Users\\u5040252\\Cloudstation\\Projects\\Mass_Spec_Data\\2016\\20160204\\20160204_cons2000.csv"}

        self.paramfile = "\\".join(self.reffiles[int(self.samples.at[s, 'cuvette'])].split("\\")[:-1]) + "\\parameters.txt"

        if os.path.isfile(self.paramfile):
            with open(self.paramfile, "r") as f:
                f32, f40 = f.read().split("\n")
                f32, f40 = float(f32), float(f40)
                return f32, f40

        else:
            df = pd.read_csv(self.reffiles[int(self.samples.at[s, 'cuvette'])], encoding='ISO-8859-1')[['time', 'Mass32', 'Mass40']]

            if df.time.max() > limit:
                df = df.loc[df.time < limit, :]

            df['d32dt'] = self.rolling_linear_reg(df.time, df['Mass32'])
            df['d40dt'] = self.rolling_linear_reg(df.time, df['Mass40'])

            # calculate slope d32dt = f(Mass32) and d40dt = f(Mass40)
            x32, y32 = np.array(df.loc[df.d32dt.notnull(), "Mass32"]), np.array(df.loc[df.d32dt.notnull(), "d32dt"])
            x40, y40 = np.array(df.loc[df.d40dt.notnull(), "Mass40"]), np.array(df.loc[df.d40dt.notnull(), "d40dt"])

            f32 = np.linalg.lstsq(x32[:, np.newaxis], y32)[0][0]
            f40 = np.linalg.lstsq(x40[:, np.newaxis], y40)[0][0]

            with open(self.paramfile, "w") as f:
                f.write("".join([str(f32), "\n", str(f40)]))

            return f32, f40

    def get_consumption_params(self, s):
        if 'cons32' in self.samples.columns and \
            self.samples.at[s, 'cons32'] is not None and \
            self.samples.at[s, 'cons32'] != 0:

            f32 = self.samples.at[s, 'cons32']
        else:
            f32 = 0

        if 'cons40' in self.samples.columns and \
            self.samples.at[s, 'cons40'] is not None and \
            self.samples.at[s, 'cons40'] != 0:

            f40 = self.samples.at[s, 'cons40']
        else:
            f40 = 0

        return f32, f40

    def denoise_data(self, s, masses=[32]):
        # get consumption parameters
        f32, f40 = self.get_consumption_params(s)

        f32 = self.consumption_params(s, limit=7200)[0] if f32 == 0 else f32
        f40 = self.consumption_params(s, limit=7200)[1] if f40 == 0 else f40

        df = self.dfs[s]

        # O2 consumption
        # correct 32 for oxygen consumption
        df.loc[:, 'dfit40dt'] = df.Mass40 * f40

        # denoising
        for m in masses:
            if ("Mass" + str(m) in df.columns) & ("d" + str(m) + "dt_d" not in df.columns):
                df["d" + str(m) + "dt_d"] = df["d" + str(m) + "dt"] - ((df["Mass" + str(m)] /
                                                                        df.Mass40) * (df.d40dt - (df.Mass40 * f40)))

                if m == 32:
                    # M32 consumption
                    df['d32dt_c'] = f32 * df.Mass32
                    df['d32dt_cd'] = df.d32dt_d - df.d32dt_c


        # df.loc[:, 'dtotalCO2dt_d'] = df.d44dt_d + df.d45dt_d + df.d46dt_d + df.d47dt_d + df.d49dt_d

        df['logE47'] = (
        np.log10(100 * df.Mass47 / (df.Mass45 + df.Mass47 + df.Mass49))).apply(self.myround)
        df['logE49'] = (
        np.log10(100 * df.Mass49 / (df.Mass45 + df.Mass47 + df.Mass49))).apply(self.myround)
        df['enrichrate47'] = self.rolling_linear_reg(df[self.timecol], df.logE47)  # (per second)
        df['enrichrate47'] = df['enrichrate47'].apply(self.myround)
        df['enrichrate49'] = self.rolling_linear_reg(df[self.timecol], df.logE49)  # (per second)
        df['enrichrate49'] = df['enrichrate49'].apply(self.myround)
        return df

    def normalize_chlorophyll(self, s):
        chl = self.samples.at[s, 'chloro_ugml'] if self.samples.at[s, 'chloro_ugml'] != 0 else 1000 # µg

        for i in ['d32dt', 'd32dt_d', 'd32dt_cd', 'd44dt', 'd45dt', 'd46dt', 'd47dt', 'd49dt', 'dtotalCO2dt','enrichrate47']:
            self.dfs[s][i + "_chl"] = self.dfs[s][i] / chl

        #for enrichment rate, we have to substract non catalytic first
        for i in ['enrichrate49']:
            cells_inj = self.dfs[s].loc[self.dfs[s].eventtype=='CELLS' ,'time'].values[0]
            nc = self.dfs[s].loc[(self.dfs[s].time >= (cells_inj-10)) & (self.dfs[s].time< cells_inj-3), i].mean()
            self.dfs[s][i + "_chl"] = (self.dfs[s][i] - nc)/chl
            print(s, i, 'NC=', nc)


    def align_event(self, df, event):
        """
        Shift time stamps to align on a given event
        :param df: dataframe
        :param event: event ID on which to align data
        :return: dataframe with a time3 column that is shift according to event
        """
        aligntime = {"BIC": 10,
                     "CELLS": 200,
                     "LIGHT ON": 350,
                     "LIGHT OFF": 600}


        if 'time2' in df.columns:
            timecol = 'time2'
        else:
            timecol = 'time'

        for s in set(df.loc[df.eventtype == event, 'echant']):
            shift = df.loc[(df.echant == s) & (df.eventtype == event), timecol].values[0] - aligntime[event]
            df.loc[(df.echant == s), 'time3'] = df.loc[(df.echant == s), timecol] - shift
            df.loc[:,'time3'] = df.time3.apply(lambda x: round(x, 1))

        return df

    def compiled_graphs(self, what, which=None, subfolder="", align="CELLS", show_noisy=False, show_denoised=True):
        """
        Create "compiled" graphs where different samples are superimposed
        :param what:
        :param which:
        :param align:
        :param show_noisy:
        :param show_denoised:
        :return:
        """
        if hasattr(what, 'lower'): what = [what]

        # check if folder exists, if not create it
        outputfolder = os.path.join(self.figures_compiled_folder, what[0], subfolder)
        print(outputfolder)
        if not os.path.isdir(os.path.join(outputfolder)):
            os.makedirs(os.path.join(outputfolder))

        if not show_noisy and not show_denoised:
            show_denoised = True

        data = self.masterdf

        if which is None:
            print("You must specify the graphs to super-impose by inputting a list of files in the variable 'which'")
        elif not os.path.isfile(os.path.join(outputfolder, self.day + '_' + "-".join(which) + "." + self.figformat)):
            # convert what and which to a list if it is a string
            if hasattr(which, 'lower'): which = [which]

            if len(set(which).difference(set(data.echant))) != 0:
                for x in (set(which).difference(set(data.echant))):
                    print(x, " is not in sample list")

            if show_denoised:
                for i, w in enumerate(what):
                    if w in ['d32dt', 'd40dt', 'd44dt', 'd45dt', 'd46dt', 'd47dt', 'd49dt']:
                        if w + "_d" in data.columns:
                            what[i] = w + "_d"
            # subset data
            data = data.loc[data.echant.isin(which), :]

            # by default align on "CELLS"
            # check that the alignment event is present for all the samples
            if len(data.loc[(data.eventtype == align), :]) == len(which):
                data = self.align_event(data, align)
            else:
                data.loc[:, 'time3'] = data[self.timecol].apply(lambda x: round(x, 1))

            # style
            style.use(self.style)

            # create figure
            plt.figure(figsize=(self.figWidth, self.figHeight))
            subset_limits = (120, 1200)

            if self.ymin == None:
                ymin = 100000
                for i, w in enumerate(what):
                    for s in which:
                        if w in self.dfs[s].columns:
                            tmp = self.dfs[s].loc[
                              (self.dfs[s][self.timecol] > subset_limits[0]) & (self.dfs[s][self.timecol] < subset_limits[1]), :]
                            ymin = min(ymin, tmp[w].min())
                self.ymin_graph = 0.9 * ymin if ymin > 0 else 1.1 * ymin
            else:
                self.ymin_graph = self.ymin


            if self.ymax == None:
                ymax = -100000
                for i, w in enumerate(what):
                    for s in which:
                        if w in self.dfs[s].columns:
                            tmp = self.dfs[s].loc[
                              (self.dfs[s][self.timecol] > subset_limits[0]) & (self.dfs[s][self.timecol] < subset_limits[1]), :]
                            ymax =max(ymax, tmp[w].max())
                self.ymax_graph = 1.1 * ymax if ymax > 0 else 0.9 * ymax
            else:
                self.ymax_graph = self.ymax


            for w in what:
                if w in data.columns:
                     # annotate
                    if self.annotate:
                        t = data.loc[
                            data.eventtype.notnull(), [self.timecol, 'eventtype', 'eventdetails', 'echant']]
                        for e in t[self.timecol]:
                            etime, etype, edetails = t.loc[t[self.timecol] == e, :][self.timecol].values[0], \
                                                     t.loc[t[self.timecol] == e, :].eventtype.values[0], \
                                                     t.loc[t[self.timecol] == e, :].eventdetails.values[0]
                            plt.axvline(etime, color=self.eventColors[etype], lw=2)

                            # highlight light event
                            if 'LIGHT ON' in t.eventtype.tolist() and 'LIGHT OFF' in t.eventtype.tolist():
                                plt.axvspan(t.loc[t.eventtype == 'LIGHT ON', [self.timecol]].values[0],
                                            t.loc[t.eventtype == 'LIGHT OFF', [self.timecol]].values[0], color='#ffffee', alpha=0.025)

                    # create plot
                    ax = data.set_index('time3').groupby(['echant'])[w].plot(lw=int(self.linewidth), label=str(w))




            # labels
            if self.title: plt.title(what[0])
            if self.xlabel: plt.xlabel('Time (s)')
            if self.ylabel: plt.ylabel("".join([what[0], "(", self.units[what[0]], ")"]))


            if self.legend:
                            # determine legend position
                legend_pos = {'logE49': 1,
                               'enrichrate49_chl':4,
                               'enrichrate49':4,
                               'd32dt_cd':2,
                               'd32dt_cd_chl':2,
                               'd45dt':3,
                               'd45dt_chl':3,
                               'd49dt':3,
                               'd49dt_chl':3,
                               }
                if what[0] in legend_pos.keys():
                    self.legend_pos = legend_pos[what[0]]
                else:
                    self.legend_pos=0
                plt.legend(loc=self.legend_pos) # 0=best, 1=top right 2: top left, 3=bot left, 4=bot right

            plt.axhline(0, color='k')
            plt.ylim(self.ymin_graph,  self.ymax_graph)



            # savefile
            plt.savefig(os.path.join(outputfolder, self.day + '_' + "-".join(which) + "." + self.figformat),
                        dpi=self.dpi)

            # close figure
            plt.close("all")
            self.log("figure", "compiled | sample:  " + "-".join(which) + ", masses: " + "-".join(what))


    def individual_graphs(self, what, ext=None, show_noisy=False, show_denoised=True):
        """ Generate graphs for each sample
        :param what:
        :param show_noisy:
        :param show_denoised:
        :return:
        """
        what = [what] if hasattr(what, 'lower') else what

        if ext is not None:
            self.figformat = ext

        if not show_noisy and not show_denoised:
            show_denoised = True

        style.use(self.style)
        outputfolder = os.path.join(self.figures_individual_folder, "_".join(what))

        # check if folder exists, if not create it
        if not os.path.isdir(outputfolder):
            os.makedirs(outputfolder)

        # if d32dt is in list, then add d32dt corrected for consumption
        if what == ['d32dt']:
            self.individual_graphs(['d32dt_cd'], show_noisy, show_denoised)
        # if what == ['Mass32']:
        #     self.individual_graphs(['Mass32_cons'], show_noisy, show_denoised)

        if show_denoised:
            for i, w in enumerate(what):
                if w in ['d32dt', 'd40dt', 'd44dt', 'd45dt', 'd46dt', 'd47dt', 'd49dt']:
                    if w + "_d" in self.masterdf.columns:
                        what[i] = w + "_d"

        for s in set(self.dfs.keys()):
            print("  ", str(s), str(what))
            self.generate_plot(outputfolder, s, what, show_noisy, show_denoised)


    def generate_plot(self, outputfolder, s, what, show_noisy=False, show_denoised=True):
        if not os.path.isfile(os.path.join(outputfolder, self.day + '_' + str(s) + "." + self.figformat)):
            self.timecol = 'time2' if 'time2' in self.dfs[s].columns else 'time'

            # create figure
            plt.figure(figsize=(self.figWidth, self.figHeight))
            ax = defaultdict(object)
            subset_limits = (100, 1200)




            if self.ymin == None:
                ymin = 100000
                for i, w in enumerate(what):
                    if w in self.dfs[s].columns:
                        tmp = self.dfs[s].loc[
                          (self.dfs[s][self.timecol] > subset_limits[0]) & (self.dfs[s][self.timecol] < subset_limits[1]), :]
                        ymin = min(ymin, tmp[w].min())
                self.ymin_graph = 0.9 * ymin if ymin > 0 else 1.1 * ymin
            else:
                self.ymin_graph = self.ymin


            if self.ymax == None:
                ymax = -100000
                for i, w in enumerate(what):
                    if w in self.dfs[s].columns:
                        tmp = self.dfs[s].loc[
                          (self.dfs[s][self.timecol] > subset_limits[0]) & (self.dfs[s][self.timecol] < subset_limits[1]), :]
                        ymax =max(ymax, tmp[w].max())
                self.ymax_graph = 1.1 * ymax if ymax > 0 else 0.9 * ymax
            else:
                self.ymax_graph = self.ymax

            for i, w in enumerate(what):
                if w in self.dfs[s].columns:
                    tmp = self.dfs[s].loc[
                          (self.dfs[s][self.timecol] > subset_limits[0]) & (self.dfs[s][self.timecol] < subset_limits[1]), :]

                # create plot
                ax[w] = self.dfs[s].set_index(self.timecol)[w].plot(lw=self.linewidth,
                                                               c=self.curveColors[w],
                                                               ls='-',
                                                               alpha=1.0,
                                                               label=str(w))

                if w in self.dfs[s].columns and len(what) == 1 and show_noisy:
                    ax[w] = self.dfs[s].set_index(self.timecol)[w].plot(lw=self.linewidth,
                                                                   c=self.curveColors[w],
                                                                   ls='-',
                                                                   alpha=0.3,
                                                                   label=str(w))

                # annotate
                if self.annotate:
                    if i == len(what)-1:
                        t = self.dfs[s].loc[
                            self.dfs[s].eventtype.notnull(), [self.timecol, 'eventtype', 'eventdetails', 'echant']]
                        for e in t[self.timecol]:
                            etime, etype, edetails = t.loc[t[self.timecol] == e, :][self.timecol].values[0], \
                                                     t.loc[t[self.timecol] == e, :].eventtype.values[0], \
                                                     t.loc[t[self.timecol] == e, :].eventdetails.values[0]
                            plt.axvline(etime, color=self.eventColors[etype], lw=2)

                            ax[w].text(x=etime - 5,
                                       y=self.ymin_graph,
                                       s=edetails,
                                       rotation=90,
                                       color=self.eventColors[etype],
                                       verticalalignment='bottom',
                                       horizontalalignment='left'
                                       )

                        # highlight light event
                        if 'LIGHT ON' in t.eventtype.tolist() and 'LIGHT OFF' in t.eventtype.tolist():
                            plt.axvspan(t.loc[t.eventtype == 'LIGHT ON', [self.timecol]].values[0],
                                        t.loc[t.eventtype == 'LIGHT OFF', [self.timecol]].values[0], color='yellow', alpha=0.025)

            # plt.legend(True)
            plt.axhline(0, color='k')

            plt.ylim(self.ymin_graph, self.ymax_graph)

            if self.title:
                plt.title(str(s))
                # plt.title(str(s) + " (" + ", ".join(what) + ")")

            # labels
            if self.xlabel:
                plt.xlabel('Time (s)')

            if self.ylabel:
                ylab = []
                for c in what:
                    if c in ["Mass44","Mass45","Mass47","Mass49","Mass46","Mass32","totalCO2"]:
                        ylab.append("[%s] (µM)" % c.strip("Mass"))
                    elif c in ["d44dt", "d44dt_d",
                               "d45dt", "d45dt_d",
                               "d46dt", "d46dt_d",
                               "d47dt", "d47dt_d",
                               "d49dt", "d49dt_d",
                               "dtotalCO2dt", "dtotalCO2dt_d",
                               "d32dt", "d32dt_d", "d32dt_cd",
                               "d44dt_d", "d44dt_d", "d44dt_d"]:
                        ylab.append("%s (µmol/L/s)" % c)
                    elif c in ["enrichrate47","enrichrate49"]:
                        ylab.append("%s (s-1)" % c)
                    elif c in ["logE47","logE49"]:
                        ylab.append(c)

                plt.ylabel(", ".join(ylab))

            if self.legend:
                plt.legend(loc=self.legend_pos)  # 1=bot left, 2=top left, 3=top right, 4=bot right

            # savefile
            plt.savefig(os.path.join(outputfolder, self.day + '_' + str(s) + "." + self.figformat), dpi=self.dpi)

            # close figure
            plt.close("all")
            self.log("figure", "individual | sample:  " + str(s) + ", mass: " + "-".join(what))

    def flux(self, where="LIGHT ON"):


        cols = ['sample', 'chl_i[µg/mL]', 'cells_vol[µL]', 'chl_f[mg/L]',
                'before49', 'after49', 'm49totCO2',
                'before45', 'after45', 'm45totCO2',
                'd32dt_on', 'd32dt_off',
                'GCU', 'GCE', 'NCU', 'GBU',
                'GCU_h_mg', 'GCE_h_mg', 'NCU_h_mg', 'GBU_h_mg']

        fluxes = pd.DataFrame(data=np.zeros((0, len(cols))), columns=cols)

        outputfolder = os.path.join(self.datafolder, self.day, "results", "flux")
        if not os.path.isdir(outputfolder):
            os.makedirs(outputfolder)

        # load data
        data = self.masterdf

        for s in set(data.loc[data.eventtype == where, 'echant']):
            print("flux", s, where)
            tmp = data.loc[data.echant == s, :]

            if where in tmp.eventtype:
                print(where, "not found")
                continue

            else:
                # align data
                tmp = self.align_event(tmp, where)

                if 'time3' in tmp.columns:
                    self.timecol = 'time3'
                elif 'time2' in tmp.columns:
                    self.timecol = 'time2'
                else:
                    self.timecol = 'time'


                t = tmp.loc[tmp.eventtype == where, self.timecol].values[0]

                if where == "LIGHT ON":
                    lon, loff = t, t + 200
                    fluxfile = os.path.join(outputfolder, self.day + "_fluxes_light.csv")
                else:
                    lon, loff = t, t + 100
                    fluxfile = os.path.join(outputfolder, self.day + "_fluxes_" + where + ".csv")


                # before light was on:
                before45 = tmp.loc[(tmp[self.timecol] > (lon - 20)) & (tmp[self.timecol] < (lon - 10)), 'd45dt'].mean()
                before49 = tmp.loc[(tmp[self.timecol] > (lon - 20)) & (tmp[self.timecol] < (lon - 10)), 'd49dt'].mean()

                # peak after light was on:
                # peak45 = tmp.query(self.timecol + ' > ' + lon + ' & ' + self.timecol + ' < ' + lon + 100)['d45dt'].min()
                # peak49 = tmp.query(self.timecol + ' > ' + lon + ' & ' + self.timecol + ' < ' + lon + 100)['d49dt'].min()
                peak45 = tmp.loc[(tmp[self.timecol] > lon) & (tmp[self.timecol] < loff), 'd45dt'].max()
                peak49 = tmp.loc[(tmp[self.timecol] > lon) & (tmp[self.timecol] < loff), 'd49dt'].min()

                # CO2 ratios
                m45totco2 = tmp.loc[tmp.eventtype == where, 'Mass45'].values[0] / tmp.loc[tmp.eventtype == where, 'totalCO2'].values[0]
                # m47totco2 = tmp.loc[tmp[self.timecol] == lon,'Mass47'].values[0] / tmp.loc[tmp[self.timecol] == lon, 'totalCO2'].values[0]
                m49totco2 = tmp.loc[tmp.eventtype == where, 'Mass49'].values[0] / tmp.loc[tmp.eventtype == where, 'totalCO2'].values[0]

                # find volume of cells injected
                if 'CELLS' in tmp.eventtype.tolist():
                    cellsvol = float(
                        re.search("[0-9]+", tmp.loc[tmp.eventtype == 'CELLS', "eventdetails"].values[0]).group())
                else:
                    cellsvol = 10
                # calculate chlorophyll in mg (from µg.µL-1 * µL cells
                '''
                Ci Vi = Cf Vf
                Cf = Ci.Vi/VF
                    Cf = mg_chl
                    Ci = chl     [µg/µL]
                    Vi = cellsvol[µL]
                    Vf = cuvette [µL]

                '''

                chl_mg_l = float(self.samples.at[s, 'chloro_ugml'])
                chl_mg_l = 1 if chl_mg_l == 0 else chl_mg_l


                # Gross CO2 uptake (GCU)
                gcu = abs((peak49 - before49) / m49totco2)  # µmol.sec-1
                gcu_h_mg = (gcu * 3600) / chl_mg_l

                # Gross CO2 Efflux (GCE)
                gce = abs((peak45 - before45) + (m45totco2 * gcu))
                gce_h_mg = (gce * 3600) / chl_mg_l

                # net Ci uptake (NCU)
                ncu = abs(tmp.loc[(tmp[self.timecol] >= (loff - 20)) & (tmp[self.timecol] <= (loff - 10)), 'd32dt'].mean() -
                          tmp.loc[(tmp[self.timecol] >= (lon - 20)) & (tmp[self.timecol] <= (lon - 10)), 'd32dt'].mean())
                ncu_h_mg = (ncu * 3600) / chl_mg_l

                # Gross bicarbonate uptake (GBU)
                gbu = abs(abs(ncu) + abs(gce) - abs(gcu))
                gbu_h_mg = (gbu * 3600) / chl_mg_l

                fluxes.loc[len(fluxes.index)] = [s, self.samples.at[s, 'chloro_ugml'], cellsvol, chl_mg_l,
                                                 before49, peak49, m49totco2,
                                                 before45, peak45, m45totco2,
                                                 tmp.loc[(tmp[self.timecol] >= (loff - 20)) & (
                                                 tmp[self.timecol] <= (loff - 10)), 'd32dt'].mean(),
                                                 tmp.loc[
                                                     (tmp[self.timecol] >= (lon - 20)) & (tmp[self.timecol] <= (lon - 10)), 'd32dt'].mean(),
                                                 gcu, gce, ncu, gbu,
                                                 gcu_h_mg, gce_h_mg, ncu_h_mg, gbu_h_mg]

                fluxes.to_csv(fluxfile, encoding=self.encoding, index=False)

        self.log("flux", "file:  " + os.path.join(outputfolder, self.day + "_fluxes_" + where + ".csv"))









