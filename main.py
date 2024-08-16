import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('AB_NYC_2019.csv')

# Data Preparation
df.dropna(subset=['last_review'], inplace=True)  # Drop rows where last_review is NaN
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')  # Convert to datetime


# Function for Neighborhood Distribution of Listings
def plot_neighborhood_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='neighbourhood_group', hue='neighbourhood_group', data=df, palette='viridis', dodge=False,
                  legend=False)
    plt.title('Distribution of Listings Across Neighborhood Groups')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Number of Listings')
    plt.xticks(rotation=45)
    plt.savefig('neighborhood_distribution.png')
    plt.show()


# Function for Price Distribution Across Neighborhoods
def plot_price_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='neighbourhood_group', y='price', hue='neighbourhood_group', data=df, palette='Set2', dodge=False,
                legend=False)
    plt.title('Price Distribution Across Neighborhood Groups')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Price')
    plt.ylim(0, 1000)  # Limiting y-axis to remove extreme outliers for better visualization
    plt.savefig('price_distribution.png')
    plt.show()


# Function for Room Type vs. Availability
def plot_room_type_vs_availability(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='neighbourhood_group', y='availability_365', hue='room_type', data=df, errorbar='sd', palette='Set3')
    plt.title('Average Availability for Each Room Type Across Neighborhoods')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Average Availability (365 days)')
    plt.legend(title='Room Type')
    plt.savefig('room_type_vs_availability.png')
    plt.show()


# Function for Correlation Between Price and Number of Reviews
def plot_price_vs_reviews(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='price', y='number_of_reviews', hue='room_type', data=df, palette='Set1', alpha=0.6)
    plt.title('Correlation Between Price and Number of Reviews')
    plt.xlabel('Price')
    plt.ylabel('Number of Reviews')
    sns.regplot(x='price', y='number_of_reviews', data=df, scatter=False, color='blue')
    plt.legend(title='Room Type')
    plt.xlim(0, 1000)  # Limiting x-axis for better visualization
    plt.ylim(0, 300)  # Limiting y-axis for better visualization
    plt.savefig('price_vs_reviews.png')
    plt.show()


# Function for Time Series Analysis of Reviews
def plot_time_series_reviews(df):
    plt.figure(figsize=(12, 6))
    df.set_index('last_review', inplace=True)
    df_grouped = df.groupby('neighbourhood_group').resample('M').number_of_reviews.mean().reset_index()
    sns.lineplot(x='last_review', y='number_of_reviews', hue='neighbourhood_group', data=df_grouped, palette='coolwarm')
    plt.title('Time Series Analysis of Reviews')
    plt.xlabel('Date')
    plt.ylabel('Average Number of Reviews')
    plt.legend(title='Neighborhood Group')
    plt.savefig('time_series_reviews.png')
    plt.show()


# Function for Price and Availability Heatmap
def plot_price_availability_heatmap(df):
    pivot_table = df.pivot_table(values='price', index='neighbourhood_group', columns='availability_365',
                                 aggfunc='mean')
    plt.figure(figsize=(14, 7))
    sns.heatmap(pivot_table, cmap='YlGnBu', cbar_kws={'label': 'Average Price'})
    plt.title('Heatmap of Price and Availability')
    plt.xlabel('Availability (365 days)')
    plt.ylabel('Neighborhood Group')
    plt.savefig('price_availability_heatmap.png')
    plt.show()


# Function for Room Type and Review Count Analysis
def plot_room_type_reviews(df):
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby(['neighbourhood_group', 'room_type']).number_of_reviews.sum().unstack().fillna(0)
    df_grouped.plot(kind='bar', stacked=True, colormap='Paired')
    plt.title('Number of Reviews for Each Room Type Across Neighborhood Groups')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Total Number of Reviews')
    plt.legend(title='Room Type')
    plt.savefig('room_type_reviews.png')
    plt.show()


# Main function to execute all plots
def main():
    plot_neighborhood_distribution(df)
    plot_price_distribution(df)
    plot_room_type_vs_availability(df)
    plot_price_vs_reviews(df)
    plot_time_series_reviews(df)
    plot_price_availability_heatmap(df)
    plot_room_type_reviews(df)


if __name__ == '__main__':
    main()
