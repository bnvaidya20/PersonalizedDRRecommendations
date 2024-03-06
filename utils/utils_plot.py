import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
color_pal = sns.color_palette()

class Plotter:
    def __init__(self, data):
        self.data = data

    def plot_power_consumption(self):
        self.data.plot(x='Date_time', y='Total_Power_Consumption', style='.',
            figsize=(16, 6),
            color=color_pal[0])
        plt.xlabel('Date')
        plt.ylabel('Power Consumption')
        plt.show()

    def plot_hourly_power_consumption(self):
        plt.figure(figsize=(10, 8)) 
        plt.scatter(self.data['hour'], self.data['Total_Power_Consumption'], alpha=0.5)
        plt.title('Power Consumption vs. Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Power Consumption')
        plt.show()

    # Investigate value with dip. For instance, Aug 2008.
    def plot_power_consumption_with_dip(self):
    
        dfj_aug = self.data.loc[(self.data['Date_time'] >= '2008-08-01') & (self.data['Date_time'] <= '2008-09-02')]

        dfj_aug.plot(x='Date_time', y='Total_Power_Consumption',style='.',
                figsize=(16, 6),
                color=color_pal[0])
        plt.xlabel('Date')
        plt.ylabel('Power Consumption')
        plt.show()

    # Visualize the seasonal power consumption.
    def plot_seasonal_power_consumption(self):

        fig, axs = plt.subplots(2, 2, figsize=(14,18), sharey=True)

        # Consumption by Hour
        sns.boxplot(data=self.data, x='hour', y='Total_Power_Consumption', ax=axs[0,0])
        axs[0,0].set_title('Hourly Power Consumption Variability')
        axs[0,0].set_xlabel('Hour of Day')
        axs[0,0].set_ylabel('Power Consumption')

        # Consumption by Day
        day_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

        sns.boxplot(data=self.data, x='day', y='Total_Power_Consumption', ax=axs[0,1])
        axs[0,1].set_title('Daily Power Consumption Variability')
        axs[0,1].set_xticklabels(day_labels)
        axs[0,1].set_ylabel('')

        # Consumption by Month
        sns.boxplot(data=self.data, x='month', y='Total_Power_Consumption', ax=axs[1,0])
        axs[1,0].set_title('Monthly Power Consumption Variability')
        axs[1,0].set_ylabel('Power Consumption')

        # Consumption by Quarter
        sns.boxplot(data=self.data, x='quarter', y='Total_Power_Consumption', ax=axs[1,1])
        axs[1,1].set_title('Quarterly Power Consumption Variability')
        axs[1,1].set_ylabel('')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.2)
        plt.show()

    # Visualize the power consumption by various submetering.
    def plot_power_consumption_by_submeter(self):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,18))

        self.data.plot(x='Date_time', y='Submeter_kitchen',style='.',color=color_pal[0], ax=axes[0,0], 
                       title='Sub metering Kitchen')
        axes[0,0].set_xlabel('')
        axes[0,0].set_ylabel('Power Consumption')
        
        self.data.plot(x='Date_time', y='Submeter_laundry',style='.',color=color_pal[1], ax=axes[0,1], 
                       title='Sub metering Laundry')
        axes[0,1].set_xlabel('')
        axes[0,1].set_ylabel('Power Consumption')
        
        self.data.plot(x='Date_time', y='Submeter_waterheat_aircon',style='.',color=color_pal[2], ax=axes[1,0], 
                       title='Sub metering Water heater and air cond')
        axes[1,0].set_xlabel('Date')
        axes[1,0].set_ylabel('Power Consumption')

        self.data.plot(x='Date_time', y='Submeter_others',style='.',color=color_pal[3], ax=axes[1,1], 
                       title='Sub metering other units')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Power Consumption')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.2)
        plt.show()

    def plot_weekend_weekday_power_consumption(self):
        self.data1= self.data.copy()
        self.data1['weekday_weekend'] = np.where(self.data1['is_weekend']==1, 'Weekend', 'Weekday')
        self.data1.boxplot(column='Total_Power_Consumption', by='weekday_weekend', figsize=(8, 6))
        plt.title('Power Consumption: Weekday vs. Weekend')
        plt.xlabel('')
        plt.ylabel('Power Consumption')
        plt.show()

    # Pivot table for heatmap
    def plot_pivot_table_heatmap(self):

        pivot_table = self.data.pivot_table(values='Total_Power_Consumption', index='day', columns='hour', aggfunc='mean')

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, cmap='YlGnBu')
        plt.title('Average Power Consumption by Hour and Day of Week')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day')
        plt.show()

    # Correlation heatmap
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))  

        corr = self.data[['Total_Power_Consumption', 'Submeter_kitchen', 'Submeter_laundry', 
                          'Submeter_waterheat_aircon', 'Submeter_others']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.xlabel('')
        plt.show()
    
    def plot_hourly_daily_mean_consumption(self):

        self.data1 = self.data.copy()

        col_req=['Date_time', 'Total_Power_Consumption', 'Submeter_kitchen', 'Submeter_laundry', 
                 'Submeter_waterheat_aircon', 'Submeter_others']

        self.data1 = self.data1[col_req]
        self.data1['Date_time'] = pd.to_datetime(self.data1['Date_time'])
        self.data1 = self.data1.set_index('Date_time')

        # Resample the data to hourly intervals and compute the mean
        hourly_data = self.data1.resample('H').mean()

        # Group by the time of day and compute the mean for each hour across all days
        daily_mean_consumption = hourly_data.groupby(hourly_data.index.time).mean()

        print(daily_mean_consumption.shape)

        daily_mean_consumption.plot(figsize=(10, 6))
        plt.title('Hourly Daily Average Power Consumption')
        plt.xlabel('Time of Day')
        plt.ylabel('Mean Power Consumption')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


    def plot_hist_daily_power_distribution(self):
        # Copy the data to avoid SettingWithCopyWarning
        self.data1 = self.data.copy()

        # Ensure 'Date_time' is in datetime format
        self.data1['Date_time'] = pd.to_datetime(self.data1['Date_time'])

        # Aggregate features ( for daily consumption)
        daily_consumption = self.data1.resample('D', on='Date_time')['Total_Power_Consumption'].sum().reset_index(name='daily_consumption')

        # Merge aggregated features back to the original DataFrame
        self.data1 = pd.merge(self.data1, daily_consumption, how='left', on='Date_time')

        # Debugging: Check the shape of the resulting DataFrame
        print(self.data1.shape)

        # Plotting
        self.data1['daily_consumption'].hist(bins=50)
        plt.title('Distribution of Daily Power Consumption')
        plt.xlabel('Daily Power Consumption')
        plt.ylabel('Frequency')
        plt.show()



class PredPlotter:

    def __init__(self, df, test_df_final):

        self.test_df_final = test_df_final

        self.df_final = df.copy()

        self.df_final = self.df_final.merge(self.test_df_final['Prediction'], how='left', left_index=True, right_index=True)

        self.df_final['Date_time'] = pd.to_datetime(self.df_final['Date_time'])
        self.df_final = self.df_final.set_index('Date_time') 

    def plot_original_prediction(self):
        # Plotting
        plt.figure(figsize=(15, 5))  
        plt.plot(self.df_final.index, self.df_final[['Total_Power_Consumption']], label='Original', color='blue')  # Plot the original values
        plt.plot(self.df_final.index, self.df_final['Prediction'], label='Predictions', color='red', linestyle='--')  # Plot the predictions

        plt.title('Original vs Predicted Values')
        plt.xlabel('Date')  
        plt.ylabel('Power Consumption')  
        plt.legend(loc='upper right')
        plt.show()

    def plot_prediction_test_period(self, test_start_date, end_date):

        test_period_df = self.df_final[test_start_date:end_date]

        # Plotting
        plt.figure(figsize=(15, 5))  

        # Plot the actual Power Consumption
        plt.plot(test_period_df.index, test_period_df['Total_Power_Consumption'], label='Actual Power Consumption', color='blue')

        # Plot the Predicted values
        plt.plot(test_period_df.index, test_period_df['Prediction'], label='Predicted Power Consumption', color='red', linestyle='--', marker='o', markersize=3)

        plt.title('Actual Power Consumption vs Predicted during Test Period')
        plt.xlabel('Date')
        plt.ylabel('Power Consumption')
        plt.legend(loc='upper right')
        plt.show()

    def plot_seasonal_prediction(self):

        # Resampling to different frequencies
        df_final_daily = self.df_final.resample('D').mean()
        df_final_monthly = self.df_final.resample('M').mean()
        df_final_quarterly = self.df_final.resample('Q').mean()


        dfs = [(df_final_daily, 'Daily'), (df_final_monthly, 'Monthly'), (df_final_quarterly, 'Quarterly')]

        # Setup the subplot grid
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True)

        for ax, (self.df_final, title) in zip(axes, dfs):
            # Plot original Power Consumption
            self.df_final['Total_Power_Consumption'].plot(ax=ax, label='Original Data')
            
            # Plot Predictions
            self.df_final['Prediction'].plot(ax=ax, style='--', label='Predictions')
            
            ax.legend()
            ax.set_title(f'Power Consumption Prediction - {title} Data')
            ax.set_xlabel('Time')
            ax.set_ylabel('Power Consumption')

        plt.tight_layout()
        plt.show()

class BasicPlotter:

    def plot_power_consumption_train_test(self, train_data, test_data, test_start_date):

        fig, ax = plt.subplots(figsize=(15, 5))
        train_data.plot(x='Date_time', y='Total_Power_Consumption', ax=ax, label='Training Set', title='Train/Test Data')
        test_data.plot(x='Date_time', y='Total_Power_Consumption', ax=ax, label='Test Set')
        ax.axvline(test_start_date, color='black', ls='--', linewidth=1)
        ax.legend(['Training Set', 'Test Set'])
        plt.show()

    def plot_actual_pred(self, y_true, y_pred):
        min_x = min(min(y_pred), min(y_true))
        max_x = max(max(y_pred), max(y_true))
        plt.scatter(y_pred, y_true)
        plt.plot([min_x,max_x], [min_x,max_x], 'r--', label = '1:1')
        plt.legend()
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        plt.show()

    @staticmethod
    def plot_feature_importances(model, feature_names):
        # Get the indices of the sorted feature importances
        sorted_idx = model.feature_importances_.argsort()
        
        # Use the provided feature names and the sorted indices to plot
        plt.barh([feature_names[i] for i in sorted_idx], model.feature_importances_[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.show()