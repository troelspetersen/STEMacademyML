import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes, load_wine

# Load datasets
DS1 = pd.read_csv('C:\\Users\\beego\\Desktop\\Visualcode_Courses\\STEMACADEMY\\Huspriser\\HousingPrices_selected.csv')

diabetes = load_diabetes()
DS2 = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

wine = load_wine()
DS3 = pd.DataFrame(data=wine.data, columns=wine.feature_names)

def main():
    st.title("Machine Learning - STEM Academy")

    # Top-level selection: Standard or Avanceret
    st.sidebar.header("Niveau")
    level = st.sidebar.radio("Vælg dit niveau:", ["Standard", "Avanceret"])

    # Dataset selection
    st.sidebar.header("Datasæt")
    dataset = st.sidebar.radio("Vælg et datasæt:", ["Huspriser", "Diabetes", "Gletsjer", "Upload dit eget datasæt"])

    # Add a description text above the displayed dataset, under the title
    st.write('Velkommen.')
    st.write("I venstre side kan du vælge niveauet for opgaverne samt hvilket datasæt du vil arbejde med. Du kan også vælge at uploade dit eget datasæt og arbejde med det. Nedenunder kan du også finde et link til vejledningen. \n Det valgte datasæt er visualiseret nedenfor.")
    

    


    # Display the selected dataset with scrolling enabled and limited to 5 rows tall
    st.subheader(f"Visualisering af {dataset}")
    if dataset == "Huspriser":
        st.dataframe(DS1, height=200, use_container_width=True)
    elif dataset == "Diabetes":
        st.dataframe(DS2, height=200, use_container_width=True)
    elif dataset == "Gletsjer":
        st.dataframe(DS3, height=200, use_container_width=True)
    elif dataset == "Upload dit eget datasæt":
        st.subheader("Upload dit eget datasæt")
        uploaded_file = st.file_uploader("Vælg en CSV-fil", type="csv")
        if uploaded_file is not None:
            user_dataset = pd.read_csv(uploaded_file)
            st.write("Her er dit datasæt:")
            st.dataframe(user_dataset, height=200, use_container_width=True)

            # Dropdown menu to select target column for regression
            target_column = st.selectbox("Vælg kolonne til regression:", user_dataset.columns, index=len(user_dataset.columns) - 1)

            # Dropdown menu to select error metric
            error_metric = st.selectbox("Vælg fejlmetrik:", ["MSE", "MAE"])

            # Button to run the model
            if st.button("Kør regression model"):
                if target_column:
                    # Prepare data
                    X = user_dataset.drop(columns=[target_column])
                    y = user_dataset[target_column]

                    # Handle missing values
                    X = X.fillna(0)
                    y = y.fillna(0)

                    # Import train_test_split before usage
                    from sklearn.model_selection import train_test_split
                    from lightgbm import LGBMRegressor
                    from sklearn.metrics import mean_squared_error

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train model
                    model = LGBMRegressor(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict and evaluate
                    y_pred = model.predict(X_test)

                    # Add predictions to the dataset
                    predictions_df = X_test.copy()
                    predictions_df[target_column] = y_test.values
                    predictions_df['Prediction'] = y_pred

                    # Display the dataset with predictions
                    st.write("Datasæt med forudsigelser:")
                    st.dataframe(predictions_df, height=200, use_container_width=True)

                    if error_metric == "MSE":
                        error = mean_squared_error(y_test, y_pred)
                        st.write(f"Mean Squared Error for regression on {target_column}: {error}")
                    elif error_metric == "MAE":
                        from sklearn.metrics import mean_absolute_error
                        error = mean_absolute_error(y_test, y_pred)
                        st.write(f"Mean Absolute Error for regression on {target_column}: {error}")

    # Add a subtle left-aligned link in the sidebar
    st.sidebar.markdown(
        """
        <div style="margin-top: 20px;">
            <a href="#" style="text-decoration: none; color: #555; font-size: 14px;">Hent vejledning</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main content 
    #Indsæt generel content her
    #Under visualisering af data


    # Content based on level and dataset
    if level == "Standard":
        if dataset == "Huspriser":
            st.subheader("Standard Niveau - Huspriser")
            st.write("Du har valgt at arbejde med huspriser. Algoritmen forudsiger prisen af et hus ud fra de resterende variable som du kan se øverst på siden.  \n Nedenfor kan du selv vælge hvordan algoritmen skal vurdere om et spørgsmål den stiller er godt eller dårligt (Se forklaring i PDF).")



            # Removed dropdown menu for selecting regression variable and set it to the last column in the dataset
            # Automatically set the target column to the last column in the dataset
            target_column = DS1.columns[-1]

            # Dropdown menu to select error metric
            error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["MSE", "MAE"])

            # Simple LGBM regression model
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error

            # Button to run the model
            if st.button("Kør regression model"):
                if target_column:
                    # Prepare data
                    X = DS1.drop(columns=[target_column])
                    y = DS1[target_column]

                    # Handle missing values
                    X = X.fillna(0)
                    y = y.fillna(0)

                    # Import train_test_split before usage
                    from sklearn.model_selection import train_test_split
                    from lightgbm import LGBMRegressor

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train model
                    model = LGBMRegressor(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict and evaluate
                    y_pred = model.predict(X_test)

                    # Add predictions to the dataset
                    predictions_df = X_test.copy()
                    predictions_df[target_column] = y_test.values
                    predictions_df['Forudsigelse'] = y_pred

                    # Display the dataset with predictions in a scrollable window
                    st.write("Datasæt med forudsigelser:")
                    st.dataframe(predictions_df, height=200, use_container_width=True)

                    if error_metric == "MSE":
                        error = mean_squared_error(y_test, y_pred)
                        st.write(f"Mean Squared Error for regression on {target_column}: {error}")
                    elif error_metric == "MAE":
                        from sklearn.metrics import mean_absolute_error
                        error = mean_absolute_error(y_test, y_pred)
                        st.write(f"Mean Absolute Error for regression on {target_column}: {error}")

            # Option to display the code
            if st.checkbox("Vis kode for regression model"):
                code = f"""
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prepare data
X = DS1.drop(columns=[target_column])
y = DS1[target_column]

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
if error_metric == "MSE":
    error = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {{error}}")
elif error_metric == "MAE":
    error = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {{error}}")
"""
                st.code(code, language="python")
        elif dataset == "Diabetes":
            st.subheader("Standard Niveau - Diabetes")
            st.write("Du har valgt at arbejde med diabetes. Algoritmen forudsiger om en person har diabetes ud fra de resterende variable.")
            # Add standard-level content for Diabetes here
        elif dataset == "Gletsjer":
            st.subheader("Standard Niveau - Gletsjer")
            st.write("Du har valgt at arbejde med gletsjere. Algoritmen forudsiger dybden af gletsjeren ud fra de resterende variable.")
            # Add standard-level content for Gletsjer here
        elif dataset == "Upload dit eget datasæt":
            st.subheader("Standard Niveau - Upload dit eget datasæt")
            st.write("Indhold for Standard Niveau og Upload dit eget datasæt.")
            # Add standard-level content for uploaded dataset here

    elif level == "Avanceret":
        if dataset == "Huspriser":
            st.subheader("Avanceret Niveau - Huspriser")
            st.write("Du har valgt at arbejde med huspriser. Nedenfor kan du vælge hvilken variabel du vil forudsige. Som udgangspunkt er dette prisen af huset.")
            # Add advanced-level content for Huspriser here
        elif dataset == "Diabetes":
            st.subheader("Avanceret Niveau - Diabetes")
            st.write("Du har valgt at arbejde med diabetes. Algoritmen forudsiger om en person har diabetes.")
            # Add advanced-level content for Diabetes here
        elif dataset == "Gletsjer":
            st.subheader("Avanceret Niveau - Gletsjer")
            st.write("Du har valgt at arbejde med gletsjere. Nedenfor kan du vælge hvilken variabel du vil forudsige. Som udgangspunkt er dette dybden af gletsjeren.")
            # Add advanced-level content for Gletsjer here
        elif dataset == "Upload dit eget datasæt":
            st.subheader("Avanceret Niveau - Upload dit eget datasæt")
            st.write("Indhold for Avanceret Niveau og Upload dit eget datasæt.")
            # Add advanced-level content for uploaded dataset here

if __name__ == "__main__":
    main()



# Spørgsmål: Skal vi lave siden så at hele vejledningen står der? Svar: Nej 
# Måske kun hvordan man kører selve koden
# fremfor introduktion til hvordan machine learning virker. Svar: Ja