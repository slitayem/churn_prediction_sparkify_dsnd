#!/usr/bin/env python3
"""
Helper functions module
"""
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col, udf
from pyspark.sql.types import StringType, DoubleType, LongType, IntegerType, DateType, TimestampType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def add_churn_column(events_df):
    """
    Add new column for customer labels Churner/Engaged
    """
    churn_users_ids = events_df.filter((events_df.page == 'Cancellation Confirmation') | \
                                       (events_df.page == 'Submit Downgrade'))\
                                                .select('userId')\
                                                .dropDuplicates()\
                                                .rdd.flatMap(lambda x : x)\
                                                .collect()
    events_df = events_df.withColumn(
        'churn', when(col("userId")\
                      .isin(list(churn_users_ids)),"Churner")\
        .otherwise("Engaged"))
    return events_df

def plot_distributions(data_counts, column, title, count_col, xlabel=None):
    """
    Plot dataframe distribution by fclass
    Args:
        data_counts: count dataframe
        column: column name to filter values of each distribution from
        fclass: column value for each of the distributions to be plot
        count_col: count values colum name
        xlabel: x axis label
    """
    plt.title(title)
    ls = {'Churner': ("-", "r"), "Engaged": ("--", "g")}
    for index, col in enumerate(data_counts[column].unique()):
        count_data = data_counts[data_counts[column] == col][count_col]
        ax = sns.kdeplot(count_data,
                    label=col, shade=True, color=ls[col][1])
        
        mean_count = count_data.mean()
        plt.ylabel("Probablity Density")
        plt.xlabel(xlabel)
        plt.axvline(mean_count, linestyle =ls[col][0], color=ls[col][1])
        # show mean values at vertical line
#         txkw = dict(size=12, color=ls[index][1], rotation=90)
#         tx = "mean: {:.2f}, std: {:.2f}".format(mean_count, count_data.std())
#         plt.text(mean_count + 1, 0.052, tx, **txkw)
    plt.grid()

    
def clean_subplots_title(grp_ax, xlabel, ylabel):
    """Clean out subplots titles
    Args:
        grp_ax: Axes list
    """
    for ax in grps.axes.flat:
        ax.set_title(ax.get_title().replace("churn = ", ""), fontsize=10)
        # This only works for the left ylabels
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)

def get_distinct_values(data_frame, column):
    """
    Get pyspake dataframe column distinct values
    Args:
        data_frame: pyspark Dataframe
        column: column name
    Return:
        array with the column unique values
    """
    col_values = data_frame.select(column) \
                .distinct().rdd.map(lambda r: r[0]).collect()
    return col_values

# plot figures one below the other
# n=len(df1.columns)
# fig,ax = plt.subplots(n,1, figsize=(6,n*2), sharex=True)
# for i in range(n):
#     plt.sca(ax[i])
#     col = df1.columns[i]
#     sns.countplot(df1[col].values)
#     ylabel(col);

def load_and_clean_data(spark, data_path):
    """
    Load and clean the dataset from data path
    Args:
        df: Spark DataFrame
    Return:
        preprocessed Spark DataFrame
    """
    df = spark.read.json(data_path)

    df = df.withColumn('event_time', F.from_unixtime(col('ts')\
                        .cast(LongType()) / 1000).cast(TimestampType()))
    df = df.withColumn('month', F.month(col('event_time')))
    df = df.withColumn('weekofyear', F.weekofyear(col('event_time')))
    df = df.withColumn('year', F.year(col('event_time')))
    df = df.withColumn('date', F.from_unixtime(col('ts') / 1000).cast(DateType()))

    df = df.filter(df.userId != '')
    return df

def plot_facet_grid(data, avg_column, xlabel, ylabel,
                    grp_col1="churn", grp_col2="userId"):
    """
    Plot field distribution for each churning and loyal users
    Args:
        data: Spark dataframe
    """
    item_df = data.groupBy([grp_col2, grp_col1]).avg(avg_column).toPandas()
    grps = sns.FacetGrid(item_df, col=grp_col1, sharey=True, palette="Blues_d",
                        size=4, aspect=1)
    grps.map(plt.hist, "avg({})".format(avg_column), alpha=.7, edgecolor='w', bins=50);
    for ax in grps.axes.flat:
        ax.set_title(ax.get_title().replace("churn = ", ""), fontsize=10)
        # This only works for the left ylabels
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)

ts_date_day = udf(lambda x: datetime.datetime.utcfromtimestamp(x / 1000), DateType())

def get_nb_songs_per_col(data, column_name, nbs_column):
    """
    Get number of songs per column `nbs_column` per column to be grouped by
    Args:
        column_name: Name of the colum to use in the group by operation
            in addition to the sessionId and userId
            e.g. level
        nbs_column: Column to get the average counts for e.g. sessionId
        data: Pyspark Dataframe containing the data
    """
    # 1. Compute average number of songs per session for each user
    # 2. Average across users for each column column_name unique value
    user_avg_songs_per_session = data.where(data.page == "NextSong") \
        .select(["userId", column_name, nbs_column]) \
        .groupby([column_name, "userId", nbs_column]).count() \
        .withColumnRenamed("count", "nb_songs_{}".format(nbs_column.capitalize())) \
        .groupby([column_name, "userId"]).mean("nb_songs_{}".format(nbs_column.capitalize())) \
        .withColumnRenamed("avg(nb_songs_{})".format(nbs_column.capitalize()),
                           "SongsPer{}".format(nbs_column.capitalize()))\
        .groupby(column_name).mean()\
        .withColumnRenamed("avg(SongsPer{})".format(nbs_column.capitalize()),
                           "avgSongsPer{}".format(nbs_column.capitalize()))
    res = pd.DataFrame(
        user_avg_songs_per_session.collect(),
        columns=user_avg_songs_per_session.columns)
    return res

def plot_days_usage(
    dataframe, groups_column,
    x_column, y_column,
    xlabel, ylabel, title="", colors_dict={},
    xticks=None):
    """
    Plot the dataframe values for each groups column unique value
    Args:
        dataframe: pandas DataFrame
        groups_column: name of the column for which the plots will be drawn
        x_column: x values column name
        y_column: z values column name
        title: plot title
        colors_dict: dictionary of groups colors
    """
    if xticks:
        plt.xticks = xticks
    for group_name in dataframe[groups_column].unique():
        group_data = dataframe[dataframe[groups_column] == group_name]

        plt.plot(group_data[x_column].values,
                group_data[y_column].values,
                label=group_name, color=colors_dict.get(group_name));
    plt.legend()
    plt.ylabel(xlabel, fontsize=10)
    plt.xlabel(ylabel, fontsize=10)
    plt.title(title)

def change_bar_width(ax, width):
    """Change bar width in barplot
    source: shorturl.at/cfJW6
    Args:
        ax: axis object
        width: new width value
    """
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - width
        patch.set_width(width)
        # recenter the bar
        patch.set_x(patch.get_x() + diff * .8)

def plot_churners_groups_count(data, title, xlabel,
                               column, groups_column, ylabel, palette="Reds_r"):
    """Plot churning and loyal users grouped by column and group_column
    Args:
        data: dataframe
        title: title string
        xlabel: x axis label
        column: column name
        groups_column: group columns list
        ylabel: y axis label
        palette: colors palette
    """
    groups_data = data.dropDuplicates(["userId", column])\
                .groupby([groups_column, column]).count()\
                .sort(groups_column).toPandas()

    ax = sns.barplot(x=groups_column, y="count",
                     hue=column, data=groups_data,
                     dodge=True,
                     palette=palette, alpha=.7)
    change_bar_width(ax, .25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best")
    plt.title(title, fontsize=14)
    return ax

def get_dataframe(dataframe, columns=None, nb_elements=5):
    """Convert a subset of a Spark Dataframe to pandas DataFrame display it
    Args:
        dataframe: sark dataframe
        nb_elements: number of rows to take from the dataframe
    """
    if not columns:
        columns = dataframe.columns
    return pd.DataFrame(dataframe.take(nb_elements), columns=columns)

def print_columns_unique_vals(dataframe, columns):
    """Print Pandas DataFrame columns unique values
    Args:
        dataframe: pandas DataFrame
        columns: array of columns names
    """
    for column in columns:
        print(dataframe.select(column).distinct())

def vector_builder(grp, columns=['count', 'date_index']):
    """Build feature vector from aggregated count
    data for each user
    Args:
        grp: ndarray of columns values
    """
    vector = np.zeros(20)
#     print(columns)
    for count, index in grp[columns].values:
        vector[int(index)] = count
    return pd.Series(vector)

account_age_in_days = udf(lambda max_date,
                          current_date: (max_date - current_date).days + 1)

def get_data_last_ndays(events_df, ndays=20, page_filter=None):
    """
    Get the last N days of the log events
    Args:
        events_df: Pyspark dataframe
        page_filter: page to filter
    """
    if page_filter:
        usage_days_df = events_df \
            .where(events_df.page == "NextSong")
    else:
        usage_days_df = events_df

    usage_days_df = usage_days_df \
        .select('userId', 'date', 'churn') \
        .groupBy('userId') \
        .agg(F.max(events_df.date), F.min(events_df.date)) \
        .withColumnRenamed('max(date)', 'last_day') \
        .withColumnRenamed('min(date)', 'first_day') \
        .withColumn("{}_days_before".format(ndays), F.date_add(col("last_day"), - ndays + 1))\
        .filter(account_age_in_days(col("last_day"), col("first_day")) >= ndays)
    return usage_days_df

def registration_days(df):
    """
    Calculates number of days between registration to last user associated event
    Args:
        df: spark DataFrame
    Return:
        df Dataframe with calculated column
    """
    last_event_df = df.groupBy("userId").max("ts").withColumnRenamed("max(ts)", "last_event")
    df = last_event_df.join(df, on="userId") \
        .withColumn("registration_days",
                    ((col("last_event") - col("registration")) / (1000 * 60 * 60 * 24)) \
        .cast(IntegerType())).select("userId", "registration_days")
    return df

def session_durations(df):
    """
    Calculates average daily and monthly session duration per user
    Args: 
        df: spark DataFrame
    Return: 
        daily session duration dataframe
    """

    daily_session_duration_df = df.groupby('userId','date','sessionId') \
            .agg(F.max('ts'), F.min('ts')) \
            .withColumn('session_duration_sec', (col('max(ts)') - col('min(ts)')) * 0.001) \
            .groupby('userId','date') \
            .avg('session_duration_sec') \
            .groupby('userId') \
            .agg(F.mean('avg(session_duration_sec)').alias('avg_daily_session_duration')) \
            .orderBy('userId', ascending=False)
    
    monthly_session_duration_df = df.groupby('userId','month','sessionId') \
            .agg(F.max('ts').alias('session_end'), F.min('ts').alias('session_start')) \
            .withColumn('session_duration_sec', (col('session_end') - col('session_start')) * 0.001) \
            .groupby('userId','month') \
            .avg('session_duration_sec') \
            .groupby('userId') \
            .agg(F.mean('avg(session_duration_sec)').alias('avg_monthly_session_duration')) \
            .orderBy('userId', ascending=False)

    return daily_session_duration_df.join(monthly_session_duration_df, on='userId')

def items_averages(df):
    """
    Calculate average number of items per session for each user(daily and monthly)
    Args:
        df: Spark DataFrame
    Return:
        daily and monthly averages DataFrame
    """
    daily_items_df = df.groupby('userId','date') \
        .max('itemInSession') \
        .groupBy('userId').avg('max(itemInSession)') \
        .withColumnRenamed('avg(max(itemInSession))', 'avg_daily_items')
    
    monthly_items_df = df.groupby('userId','month') \
        .max('itemInSession') \
        .groupBy('userId').avg('max(itemInSession)') \
        .withColumnRenamed('avg(max(itemInSession))', 'avg_monthly_items')
    
    return daily_items_df.join(monthly_items_df, on='userId')

def impute_missing_values(spark, df, column_name, original_df):
    """Set the number of errors to zero for the users not having any error event
    Args:
        df: Spark dataFrame
        original_df: spark dataframe with the rest of userIds\
            having missing values
    Return:
        df with missing values set to zero
    """
    df_usersids = list(df.select('userId') \
                           .dropDuplicates().rdd.flatMap(lambda x : x) \
                           .collect())
    missing_ids = original_df.filter(~original_df.userId.isin(df_usersids)).select('userId') \
                                    .dropDuplicates().rdd.flatMap(lambda x : x).collect()

    other_users_df = spark.createDataFrame([(userId, 0) for userId in missing_ids], ['userId', column_name])
    df = df.union(other_users_df)
    return df

def evaluate_model(model, start, end, validation_df, model_description, results_dict=None):
    """Evaluate Model using F1 and AUC metrics
    Args:
        model: Model Object
        start: training start time
        end: training end time
        validation_df: Validation Data Pandas DataFrame
        model_description: model description
        results_dict: Model results dictionary.
            Can be used to compare several models performance
    """
    if not results_dict:
        results_dict = {}
    results_lr = model.transform(validation_df)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
    evaluator2 = BinaryClassificationEvaluator(labelCol="label")
    results_dict[model_description] = {
        "F1-Score": round(evaluator.evaluate(results_lr, {evaluator.metricName: "f1"}), 4),
        "AUC": round(evaluator2.evaluate(results_lr, {evaluator2.metricName: "areaUnderROC"}), 4),
        "Training Time": end - start
    }
    print("{} Evaluation:".format(model_description))
    print('AUC: {}'.format(results_dict[model_description]["AUC"]))
    print('F1-Score: {}'.format((results_dict[model_description]["F1-Score"])))
    return results_dict

def get_gridsearch_resuts(cv_model):
    """Get GridSearch results
    """
    scores = cv_model.avgMetrics
    params = [{
        param.name: value for param, value in m_.items()}
        for m_ in cv_model.getEstimatorParamMaps()]
    params_df = pd.DataFrame(params)
    params_df['AUC score'] = scores
    return params_df

def get_feature_importance(model, features_names):
    """Get features importance from Pyspark model objet
    Args:
        model: model object
        features_names: list of features names
    Return:
        DataFrame of features importances
    """
    f_importances = model.stages[-1].featureImportances
    f_importances = [f_importances[index] for index in range(len(f_importances))]
    return pd.DataFrame(
        {"feature": features_names,
        "importance": importances_list, }
    ).sort_values('importance', ascending = False)


def plot_df_barplot(df):
    """Plot bar plot from input dataframe
    Args:
        df: Pandas DataFrame

    """
    #plt.rcParams['figure.figsize'] = (9,6)
    #plt.subplots_adjust(left=0.20, right=0.9, top=0.95, bottom=0.15)
    sns.barplot(data = df, y = "feature", x ="importance",
               palette = 'Blues_r', zorder=2, alpha=.7);
    plt.grid(axis = 'x', linestyle = '--', zorder=0)
    plt.title("Feature Importance")
    #plt.ylabel("");