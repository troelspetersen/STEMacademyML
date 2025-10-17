import streamlit as st
import pandas as pd

st.set_page_config(page_title="Avanceret Niveau", page_icon="🚀")

# Cached data loading functions
@st.cache_data
def load_housing_data():
    """Load housing prices dataset with caching"""
    return pd.read_csv('data/HousingPrices_selected.csv')

@st.cache_data
def load_diabetes_data():
    """Load diabetes dataset with caching"""
    return pd.read_csv('data/diabetes_data_rounded.csv')

@st.cache_data
def load_gletsjer_data():
    """Load gletsjer dataset with caching"""
    return pd.read_csv('data/gletsjer_data_rounded.csv')

def main():

    # Load datasets using cached functions
    DS1 = load_housing_data()
    DS2 = load_diabetes_data()
    DS3 = load_gletsjer_data()

    # Dataset selection
    st.sidebar.header("Datasæt")
    dataset = st.sidebar.radio("Vælg et datasæt:", ["Huspriser", "Diabetes", "Gletsjer", "Upload dit eget datasæt"])

    st.title(f"🚀 Avanceret Niveau")

    # Add description
    st.write("Vælg et datasæt for at begynde.")
    st.write('Under "Avanceret" har du mere mulighed for at lege med modellen, skrue på flere parametre og bedre analysere dens performance. Prøv denne af hvis du føler dig komfortabel med "Standard".')

    # Display the selected dataset with scrolling enabled and limited to 5 rows tall
    if dataset != "Upload dit eget datasæt":
        st.subheader(f"Visualisering af {dataset}")
    else:
        st.subheader(f"Visualisering af dit eget datasæt")
    if dataset == "Huspriser":
        st.dataframe(DS1, height=200, use_container_width=True)
    elif dataset == "Diabetes":
        st.dataframe(DS2, height=200, use_container_width=True)
    elif dataset == "Gletsjer":
        st.dataframe(DS3, height=200, use_container_width=True)
    elif dataset == "Upload dit eget datasæt":
        #st.subheader("Avanceret niveau - Upload dit eget datasæt")
        st.write('Upload et datasæt og prøv selv at få modellen til at forudsige en af dine kolonner. Du kan selv vælge hvilken slags model du vil bruge.')
        uploaded_file = st.file_uploader("Vælg en CSV-fil", type="csv")
        if uploaded_file is not None:
            user_dataset = pd.read_csv(uploaded_file)
            st.write("Her er dit datasæt:")
            st.dataframe(user_dataset, height=200, use_container_width=True)

            # Choose between regression and classification
            model_type = st.radio("Vælg model type:", ["Regression", "Klassifikation"])
            
            # Dropdown menu to select target column
            target_column = st.selectbox("Vælg kolonne til forudsigelse:", user_dataset.columns, index=len(user_dataset.columns) - 1)

            if model_type == "Regression":
                # Allow user to select columns for regression (excluding the target column)
                available_columns = [col for col in user_dataset.columns if col != target_column]
                selected_columns = st.multiselect(
                    "Hvilke variable vil du bruge til at forudsige den med?", 
                    available_columns, 
                    default=available_columns
                )

                # Dropdown menu to select error metric
                error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

                # Button to run the model
                if st.button("Kør regression model"):
                    # Ensure at least one column is selected
                    if not selected_columns:
                        st.error("Vælg mindst én kolonne til regression.")
                    else:
                        # Prepare data using only selected columns
                        X = user_dataset[selected_columns]
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

                        # Train model with selected loss function
                        if error_metric == "Kvadreret fejl":
                            model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)
                        elif error_metric == "Absolut fejl":
                            model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)
                        model.fit(X_train, y_train)

                        # Predict and evaluate
                        y_pred = model.predict(X_test)

                        # Add predictions to the dataset
                        predictions_df = X_test.copy()
                        predictions_df[target_column] = y_test.values
                        predictions_df['Forudsigelse'] = y_pred

                        # Display the dataset with predictions
                        st.write("Datasæt med forudsigelser:")
                        st.dataframe(predictions_df, height=200, use_container_width=True)

                        if error_metric == "Kvadreret fejl":
                            error = mean_squared_error(y_test, y_pred)
                            st.write(f"Kvadreret fejl for regression på {target_column}: {error}")
                        elif error_metric == "Absolut fejl":
                            from sklearn.metrics import mean_absolute_error
                            error = mean_absolute_error(y_test, y_pred)
                            st.write(f"Absolut fejl for regression på {target_column}: {error}")

                        # Rigtig vs forudsagt plot
                        st.subheader("Rigtige vs forudsagte værdier værdier")
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
                        
                        # Perfect prediction line
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfekt foudsigelse')
                        
                        ax.set_xlabel('Rigtige værdier')
                        ax.set_ylabel('Foroudsagte værdier')
                        ax.set_title('Rigtige vs forudsagte værdier')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)

                        # Histogram of residuals
                        st.subheader("Histogram over residualer")
                        residuals = y_test - y_pred
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
                        ax.set_xlabel('Residualer (Rigtig - Forudsagt)')
                        ax.set_ylabel('Hyppighed')
                        ax.set_title('Fordeling af residualer')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Residual statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Gennemsnit af residualer", f"{np.mean(residuals):.3f}")
                        with col2:
                            st.metric("Usikkerhed på residualer", f"{np.std(residuals):.3f}")
                        with col3:
                            st.metric("Max |residualer|", f"{np.max(np.abs(residuals)):.3f}")
            
            elif model_type == "Klassifikation":
                # Allow user to select columns for classification (excluding the target column)
                available_columns = [col for col in user_dataset.columns if col != target_column]
                selected_columns = st.multiselect(
                    "Vælg kolonner til klassifikation:", 
                    available_columns, 
                    default=available_columns
                )

                # Button to run the model
                if st.button("Kør klassifikations model"):
                    # Ensure at least one column is selected
                    if not selected_columns:
                        st.error("Vælg mindst én kolonne til klassifikation.")
                    else:
                        # Prepare data using only selected columns
                        X = user_dataset[selected_columns]
                        y = user_dataset[target_column]

                        # Handle missing values
                        X = X.fillna(0)
                        y = y.fillna(0)

                        # Train-test split
                        from sklearn.model_selection import train_test_split
                        from lightgbm import LGBMClassifier
                        from sklearn.metrics import accuracy_score

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Train a classification model using LGBMClassifier
                        model = LGBMClassifier(random_state=42)
                        model.fit(X_train, y_train)

                        # Make predictions
                        y_pred = model.predict(X_test)

                        # Add predictions to the dataset
                        predictions_df = X_test.copy()
                        predictions_df[target_column] = y_test.values
                        predictions_df['Forudsigelse'] = y_pred

                        # Display the dataset with predictions
                        st.write("Datasæt med forudsigelser:")
                        st.dataframe(predictions_df, height=200, use_container_width=True)

                        # Show accuracy score
                        score = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy for klassifikation på {target_column}: {score:.4f}")

                        # ROC Curve visualization
                        st.subheader("📈 ROC Curve")
                        from sklearn.metrics import roc_curve, auc
                        import matplotlib.pyplot as plt

                        # Get prediction probabilities
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        # Create ROC plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC Curve for Classification')
                        ax.legend(loc="lower right")
                        ax.grid(True)
                        
                        st.pyplot(fig)
                        st.write(f"AUC Score: {roc_auc:.4f}")

    # Add a download link for guidance PDF in the sidebar
    import os
    import base64
    pdf_path = 'data/vejledning.pdf'  # Put your PDF file here
    
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Create base64 encoded PDF for download
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        
        # Create a download link that looks exactly like the original
        st.sidebar.markdown(
            f"""
            <div style="margin-top: 20px;">
                <a href="data:application/pdf;base64,{b64_pdf}" download="vejledning.pdf" style="text-decoration: none; color: #555; font-size: 14px;">Hent vejledning</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="margin-top: 20px;">
                <span style="color: #999; font-size: 14px;">Hent vejledning (PDF ikke fundet)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Content based on dataset - Avanceret level
    if dataset == "Huspriser":
        st.subheader("Avanceret Niveau - Huspriser")
        st.write("Algoritmen forudsiger prisen af et hus ud fra de resterende variable som du kan se øverst på siden.  \n Nedenfor kan du selv vælge hvordan algoritmen skal vurdere om et spørgsmål den stiller er godt eller dårligt (Se forklaring i PDF).")
        
        target_column = st.selectbox("Hvilken variabel vil du forudsige?", DS1.columns, index=len(DS1.columns) - 1)

        # Allow user to select columns for regression (excluding the target column)
        available_columns = [col for col in DS1.columns if col != target_column]
        selected_columns = st.multiselect(
            "Hvilke variable vil du bruge til at forudsige den med?", 
            available_columns, 
            default=available_columns
        )

        # Error metric selection
        error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

        # Button to run the model
        if st.button("Kør regression model"):
            # Ensure at least one column is selected
            if not selected_columns:
                st.error("Vælg mindst én kolonne til regression.")
            else:
                # Prepare data using only selected columns
                X = DS1[selected_columns]
                y = DS1[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Train-test split
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMRegressor
                from sklearn.metrics import mean_squared_error

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a regression model using LGBMRegressor with selected objective
                if error_metric == "Kvadreret fejl":
                    model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)
                elif error_metric == "Absolut fejl":
                    model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                if error_metric == "Kvadreret fejl":
                    error = mean_squared_error(y_test, y_pred)
                    st.write(f"Kvadreret fejl for regression på {target_column}: {error} kr.")
                elif error_metric == "Absolut fejl":
                    from sklearn.metrics import mean_absolute_error
                    error = mean_absolute_error(y_test, y_pred)
                    st.write(f"Absolut fejl for regression på {target_column}: {error} kr.")

                # Rigtig vs forudsagt plot
                st.subheader("Rigtige vs forudsagte værdier værdier")
                import matplotlib.pyplot as plt
                import numpy as np
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfekt foudsigelse')
                
                ax.set_xlabel('Rigtige værdier')
                ax.set_ylabel('Foroudsagte værdier')
                ax.set_title('Rigtige vs forudsagte værdier')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

                # Histogram of residuals
                st.subheader("Histogram over residualer")
                residuals = y_test - y_pred
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
                ax.set_xlabel('Residualer (Rigtig - Forudsagt)')
                ax.set_ylabel('Hyppighed')
                ax.set_title('Fordeling af residualer')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Residual statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gennemsnit af residualer", f"{np.mean(residuals):.3f}")
                with col2:
                    st.metric("Usikkerhed på residualer", f"{np.std(residuals):.3f}")
                with col3:
                    st.metric("Max |residualer|", f"{np.max(np.abs(residuals)):.3f}")

    elif dataset == "Diabetes":
        st.subheader("Avanceret Niveau - Diabetes")
        st.write("Nedenfor kan du vælge hvilke variable du vil bruge til at forudsige diabetes. Ikke alle af disse er nødvendigvis lige relevante for at forudsige resultatet.")
        
        # Target column is fixed for diabetes
        target_column = 'Diabetes'

        # Allow user to select columns for classification (excluding the target column)
        available_columns = [col for col in DS2.columns if col != target_column]
        selected_columns = st.multiselect(
            "Vælg kolonner til klassifikation:", 
            available_columns, 
            default=available_columns
        )

        # Button to run the model
        if st.button("Kør klassifikations model"):
            # Ensure at least one column is selected
            if not selected_columns:
                st.error("Vælg mindst én kolonne til klassifikation.")
            else:
                # Prepare data using only selected columns
                X = DS2[selected_columns]
                y = DS2[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Train-test split
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMClassifier
                from sklearn.metrics import accuracy_score

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a classification model using LGBMClassifier
                model = LGBMClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                # Show accuracy score
                score = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy for klassifikation på {target_column}: {score:.4f}")

                # ROC Curve visualization
                st.subheader("📈 ROC Curve")
                from sklearn.metrics import roc_curve, auc
                import matplotlib.pyplot as plt
    
                
                # Get prediction probabilities
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve for Diabetes Classification')
                ax.legend(loc="lower right")
                ax.grid(True)
                
                st.pyplot(fig)
                st.write(f"AUC Score: {roc_auc:.4f}")

                # Histogram of predicted probabilities by actual class
                st.subheader("Fordeling af syge og raske patienter som funktion af modellens forudsigelse")
                import numpy as np
                
                # Split probabilities by actual class (ground truth)
                prob_no_diabetes = y_pred_proba[y_test == 0]  # Actually no diabetes
                prob_diabetes = y_pred_proba[y_test == 1]     # Actually diabetes
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(prob_no_diabetes, bins=30, alpha=0.6, color='lightblue', 
                       label=f'Ingen diabetes (n={len(prob_no_diabetes)})', edgecolor='black')
                ax.hist(prob_diabetes, bins=30, alpha=0.6, color='lightcoral', 
                       label=f'Diabetes (n={len(prob_diabetes)})', edgecolor='black')
                ax.set_xlabel('Forudsagt sandsynlighed for diabetes')
                ax.set_ylabel('Antal')
                ax.set_title('Fordeling af forudsagte sandsynligheder efter faktisk diagnose')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Beslutningsgrænse (0.5)')
                ax.legend()
                
                st.pyplot(fig)
                
                # Show some statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Antal korrekt klassificeret - Ingen diabetes", f"{len(prob_no_diabetes[prob_no_diabetes < 0.5])}")
                with col2:
                    st.metric("Antal korrekt klassificeret - Diabetes", f"{len(prob_diabetes[prob_diabetes >= 0.5])}")
        
    elif dataset == "Gletsjer":
        st.subheader("Avanceret Niveau - Gletsjer")
        st.write("Algoritmen forudsiger dybden af gletsjeren ud fra de resterende variable som du kan se øverst på siden.  \n Nedenfor kan du selv vælge hvordan algoritmen skal vurdere om et spørgsmål den stiller er godt eller dårligt (Se forklaring i PDF).")
        
        target_column = st.selectbox("Hvilken variabel vil du forudsige?", DS3.columns, index=len(DS3.columns) - 1)

        # Allow user to select columns for regression (excluding the target column)
        available_columns = [col for col in DS3.columns if col != target_column]
        selected_columns = st.multiselect(
            "Hvilke variable vil du bruge til at forudsige den med?", 
            available_columns, 
            default=available_columns
        )

        # Error metric selection
        error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

        # Button to run the model
        if st.button("Kør regression model"):
            # Ensure at least one column is selected
            if not selected_columns:
                st.error("Vælg mindst én kolonne til regression.")
            else:
                # Prepare data using only selected columns
                X = DS3[selected_columns]
                y = DS3[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Train-test split
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMRegressor
                from sklearn.metrics import mean_squared_error

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a regression model using LGBMRegressor with selected objective
                if error_metric == "Kvadreret fejl":
                    model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)
                elif error_metric == "Absolut fejl":
                    model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                if error_metric == "Kvadreret fejl":
                    error = mean_squared_error(y_test, y_pred)
                    st.write(f"Kvadreret fejl for regression på {target_column}: {error} kr.")
                elif error_metric == "Absolut fejl":
                    from sklearn.metrics import mean_absolute_error
                    error = mean_absolute_error(y_test, y_pred)
                    st.write(f"Absolut fejl for regression på {target_column}: {error} kr.")

                # Rigtig vs forudsagt plot
                st.subheader("Rigtig vs forudsagt plot")
                import matplotlib.pyplot as plt
                import numpy as np
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfekt foudsigelse')
                
                ax.set_xlabel('Rigtige værdier')
                ax.set_ylabel('Forudsagte værdier')
                ax.set_title('Rigtige vs forudsagte værdier')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

                # Histogram of residuals
                st.subheader("📊 Histogram over residualer")
                residuals = y_test - y_pred
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
                ax.set_xlabel('Residualer (Rigtig - Forudsagt)')
                ax.set_ylabel('Hyppighed')
                ax.set_title('Fordeling af residualer')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Residual statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gennemsnit af residualer", f"{np.mean(residuals):.3f}")
                with col2:
                    st.metric("Usikkerhed på residualer", f"{np.std(residuals):.3f}")
                with col3:
                    st.metric("Max |residualer|", f"{np.max(np.abs(residuals)):.3f}")
        
    # elif dataset == "Upload dit eget datasæt":
    #     st.subheader("Avanceret Niveau - Upload dit eget datasæt")
    #     st.write("Indhold for Avanceret Niveau og Upload dit eget datasæt.")
    #     # Add advanced-level content for uploaded dataset here

if __name__ == "__main__":
    main()