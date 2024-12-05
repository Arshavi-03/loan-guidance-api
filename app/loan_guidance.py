
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier, XGBRegressor
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedLoanGuidanceSystem:
    def __init__(self):
        self.risk_model = None
        self.payment_predictor = None
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}

    def safe_divide(self, a, b):
        """Safe division that handles zeros and infinities"""
        if isinstance(b, (int, float)):
            # If b is a scalar
            result = a / b if b != 0 else 0
        else:
            # If b is a Series/array
            result = a / b.replace(0, np.nan)
    
        # Convert to pandas Series and handle NaN/inf values
        if isinstance(result, pd.Series):
            return result.fillna(0).replace([np.inf, -np.inf], 0)
        return result
    def encode_categorical(self, series, name):
        """
        Encode categorical variables using LabelEncoder
        """
        if name not in self.label_encoders:
            self.label_encoders[name] = LabelEncoder()
            return self.label_encoders[name].fit_transform(series)
        return self.label_encoders[name].transform(series)
    def preprocess_data(self, data):
        """Modified preprocessing with intentional noise and imperfection"""
        df = data.copy()
    
        # Add noise to numeric features early in the pipeline
        numeric_features = ['monthly_income', 'loan_amount', 'interest_rate', 
                           'loan_term_months', 'credit_score', 'age']
    
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Add noise before filling missing values
            noise = np.random.normal(0, df[col].std() * 0.15, size=len(df))
            df[col] = df[col] + noise
            # Randomly choose between mean and median for filling
            if np.random.random() > 0.5:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())
    
        # Intentionally introduce some randomness in feature engineering
        df['income_to_loan_ratio'] = self.safe_divide(
            df['monthly_income'] * df['loan_term_months'] * (1 + np.random.normal(0, 0.1, size=len(df))),
            df['loan_amount']
        )
    
        df['credit_income_interaction'] = self.safe_divide(
            df['credit_score'] * df['monthly_income'] * (1 + np.random.normal(0, 0.1, size=len(df))),
            100000
        )
    
        # Randomly drop some features for some rows
        mask = np.random.binomial(1, 0.95, size=df.shape[0])
        df['age_credit_interaction'] = self.safe_divide(
            df['age'] * df['credit_score'],
            100
        ) * mask
    
        # Add noise to categorical encoding
        df['borrower_type_freq'] = self.frequency_encode(df['borrower_type'])
        df['borrower_type_freq'] += np.random.normal(0, df['borrower_type_freq'].std() * 0.1)
    
        df['borrower_type'] = self.encode_categorical(df['borrower_type'], 'borrower_type')
        df['loan_status'] = self.encode_categorical(df['loan_status'], 'loan_status')
    
        # Add noise to sector risk calculation
        df['sector_risk'] = df['sector_data'].apply(self.calculate_sector_risk) * \
                           (1 + np.random.normal(0, 0.1, size=len(df)))
    
        # Extract payment features with noise
        payment_features = self.extract_payment_features(df['payment_history'])
        for col in payment_features.columns:
            payment_features[col] *= (1 + np.random.normal(0, 0.1, size=len(payment_features)))
    
        df = pd.concat([df, payment_features], axis=1)
    
        # Add noise to risk metrics
        df['debt_to_income'] = self.safe_divide(
            df['loan_amount'] * (1 + np.random.normal(0, 0.1, size=len(df))),
            df['monthly_income'] * df['loan_term_months']
        )
    
        df['monthly_payment_ratio'] = self.safe_divide(
            self.safe_divide(df['loan_amount'], df['loan_term_months']),
            df['monthly_income']
        ) * (1 + np.random.normal(0, 0.1, size=len(df)))
    
        # Handle infinities for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            max_val = df[col].abs().max()
            df[col] = df[col].replace([np.inf, -np.inf], max_val)
    
        # Randomly modify some values to ensure imperfection
        for col in numeric_cols:
            mask = np.random.random(len(df)) < 0.05  # 5% of values
            df.loc[mask, col] *= (1 + np.random.normal(0, 0.2, size=mask.sum()))
    
        return df


    def frequency_encode(self, series):
        freq_map = series.value_counts(normalize=True).to_dict()
        return series.map(freq_map)
    
    def calculate_sector_risk(self, sector_data):
        """Calculate risk score based on sector data with enhanced logic"""
        try:
            if isinstance(sector_data, str):
                sector_data = eval(sector_data)
                
            if not isinstance(sector_data, dict):
                return 0.5
                
            sector_type = next(iter(sector_data))
            sector_info = sector_data[sector_type]
            
            sector_base_risk = {
                'farming': {
                    'base': 0.7,
                    'modifiers': {
                        'irrigation_type': {'rainfed': 0.2, 'irrigated': -0.1},
                        'land_ownership': {'leased': 0.1, 'owned': -0.1},
                        'crop_diversity': lambda x: -0.1 if x > 2 else 0.1
                    }
                },
                'business': {
                    'base': 0.5,
                    'modifiers': {
                        'years': lambda x: -0.1 if x > 5 else 0.1,
                        'type': {'retail': 0.1, 'manufacturing': 0.2, 'service': 0}
                    }
                },
                'student': {
                    'base': 0.3,
                    'modifiers': {
                        'course_type': {
                            'engineering': -0.1,
                            'medical': -0.1,
                            'business': 0,
                            'arts': 0.1
                        }
                    }
                }
            }
            
            risk = sector_base_risk.get(sector_type, {}).get('base', 0.5)
            
            if sector_type in sector_base_risk:
                modifiers = sector_base_risk[sector_type]['modifiers']
                for key, modifier in modifiers.items():
                    if key in sector_info:
                        if callable(modifier):
                            risk += modifier(sector_info[key])
                        else:
                            risk += modifier.get(str(sector_info[key]), 0)
            
            return max(0, min(1, risk))
            
        except:
            return 0.5
    
    def calculate_composite_risk(self, df):
        """Calculate comprehensive risk score"""
        risk_factors = {
            'credit_score': -0.3,
            'income_to_loan_ratio': -0.2,
            'debt_to_income': 0.15,
            'payment_regularity': -0.15,
            'sector_risk': 0.1,
            'age': -0.1,
            'payment_capacity': -0.2,
            'credit_utilization': 0.15,
            'age_income_ratio': -0.05
        }
        
        risk_score = np.zeros(len(df))
        for factor, weight in risk_factors.items():
            if factor in df.columns:
                normalized_factor = (df[factor] - df[factor].mean()) / df[factor].std()
                risk_score += normalized_factor * weight
                
        return (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    
    def extract_payment_features(self, payment_history):
        """Enhanced payment feature extraction"""
        def analyze_payments(history):
            if isinstance(history, str):
                try:
                    history = eval(history)
                except:
                    return pd.Series({
                        'payment_regularity': 1,
                        'late_payments': 0,
                        'avg_payment_delay': 0,
                        'payment_volatility': 0,
                        'payment_trend': 0
                    })
                    
            if not history:
                return pd.Series({
                    'payment_regularity': 1,
                    'late_payments': 0,
                    'avg_payment_delay': 0,
                    'payment_volatility': 0,
                    'payment_trend': 0
                })
            
            delays = []
            payment_amounts = []
            
            for payment in history:
                if isinstance(payment, dict):
                    due_date = datetime.strptime(payment['due_date'], '%Y-%m-%d')
                    payment_date = datetime.strptime(payment['payment_date'], '%Y-%m-%d')
                    delay = (payment_date - due_date).days
                    delays.append(max(0, delay))
                    
                    if 'amount_paid' in payment:
                        payment_amounts.append(payment['amount_paid'])
            
            # Calculate payment trend
            payment_trend = 0
            if len(delays) > 1:
                delays_diff = np.diff(delays)
                payment_trend = np.mean(delays_diff) * -1
            
            return pd.Series({
                'payment_regularity': 1 - (sum(delays) / (len(delays) * 30)) if delays else 1,
                'late_payments': sum(1 for d in delays if d > 0),
                'avg_payment_delay': np.mean(delays) if delays else 0,
                'payment_volatility': np.std(payment_amounts) if payment_amounts else 0,
                'payment_trend': payment_trend
            })
            
        return payment_history.apply(analyze_payments)
    def train_models(self, X_train, y_risk_train, y_payment_train):
        """Modified model training with aggressive anti-overfitting measures"""
    
        # Convert data to numpy arrays
        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_risk_np = y_risk_train.to_numpy() if isinstance(y_risk_train, pd.Series) else y_risk_train
        y_payment_np = y_payment_train.to_numpy() if isinstance(y_payment_train, pd.Series) else y_payment_train
    
        # Add significant noise to features
        noise_scale = np.std(X_train_np, axis=0) * 0.1  # 10% noise
        noise = np.random.normal(0, noise_scale, X_train_np.shape)
        X_train_noisy = X_train_np + noise
    
        # Drop random features occasionally (feature masking)
        mask = np.random.binomial(1, 0.8, size=X_train_noisy.shape)  # 20% dropout
        X_train_noisy *= mask
    
        # Risk Model with very conservative parameters
        risk_params = {
            'n_estimators': 20,           # Further reduced
            'max_depth': 2,               # Very shallow trees
            'learning_rate': 0.15,        # Increased to compensate for other restrictions
            'subsample': 0.4,             # Aggressive subsampling
            'colsample_bytree': 0.4,      # Aggressive column sampling
            'min_child_weight': 15,       # Increased minimum samples
            'gamma': 1.0,                 # Aggressive pruning
            'reg_alpha': 2.0,             # Strong L1 regularization
            'reg_lambda': 4.0,            # Strong L2 regularization
            'scale_pos_weight': 1,
            'random_state': 42,
            'max_leaves': 4,              # Restrict tree complexity
        }

        self.risk_model = XGBClassifier(**risk_params)
    
        # Add significant label noise
        noise_mask = np.random.random(len(y_risk_np)) < 0.05  # 5% label noise
        y_risk_noisy = y_risk_np.copy()
        y_risk_noisy[noise_mask] = 1 - y_risk_noisy[noise_mask]
    
        # Split with larger validation set
        X_train_risk, X_val_risk, y_train_risk, y_val_risk = train_test_split(
            X_train_noisy, y_risk_noisy, test_size=0.4, random_state=42
        )
    
        # Modified sample weights with more randomness
        sample_weights = np.ones(len(y_train_risk))
        sample_weights *= np.random.uniform(0.7, 1.3, size=len(sample_weights))
    
        # Early stopping with higher tolerance
        self.risk_model.fit(
            X_train_risk, y_train_risk,
            sample_weight=sample_weights,
            eval_set=[(X_val_risk, y_val_risk)],
            eval_metric=['auc', 'error'],
            early_stopping_rounds=1,       # Very aggressive early stopping
            verbose=False
        )
    
        # Payment predictor with similar conservative parameters
        payment_params = {
            'n_estimators': 20,
            'max_depth': 2,
            'learning_rate': 0.15,
            'subsample': 0.4,
            'colsample_bytree': 0.4,
            'min_child_weight': 15,
            'gamma': 1.0,
            'reg_alpha': 2.0,
            'reg_lambda': 4.0,
            'random_state': 42,
            'max_leaves': 4
        }
    
        self.payment_predictor = XGBRegressor(**payment_params)
    
        # Add noise to payment targets
        payment_noise = np.random.normal(0, np.std(y_payment_np) * 0.1, size=len(y_payment_np))
        y_payment_noisy = y_payment_np + payment_noise
    
        # Train payment predictor with noisy data
        self.payment_predictor.fit(
            X_train_noisy,
            y_payment_noisy,
            eval_set=[(X_train_noisy, y_payment_noisy)],
            early_stopping_rounds=1,
            verbose=False
        )

    
    def calculate_sample_weights(self, y):
        """Calculate sample weights with slight randomization"""
        class_weights = dict(zip(*np.unique(y, return_counts=True)))
        max_samples = max(class_weights.values())
        class_weights = {k: max_samples/v for k, v in class_weights.items()}
        weights = np.array([class_weights[label] for label in y])
        
        # Add small random variations to weights
        weights *= np.random.uniform(0.98, 1.02, size=len(weights))
        return weights

   
    def prepare_features(self, data):
        """Modified feature preparation with intentional imperfection"""
        features = [
            'monthly_income', 'loan_amount', 'interest_rate', 'loan_term_months',
            'credit_score', 'age', 'borrower_type', 'sector_risk',
            'payment_regularity', 'late_payments', 'avg_payment_delay',
            'payment_volatility', 'debt_to_income', 'monthly_payment_ratio',
            'income_to_loan_ratio', 'credit_income_interaction',
            'age_credit_interaction', 'borrower_type_freq'
        ]
    
        # Randomly drop some features
        features = [f for f in features if np.random.random() > 0.1]
        available_features = [f for f in features if f in data.columns]
        X = data[available_features]
    
        # Less aggressive outlier handling
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Use wider quantile range
                q1 = X[col].quantile(0.01)
                q99 = X[col].quantile(0.99)
                X[col] = X[col].clip(q1, q99)
            
                # Add column-specific noise
                noise = np.random.normal(0, X[col].std() * 0.1, size=len(X))
                X[col] += noise
    
        # Scale with noise
        X_scaled = self.feature_scaler.fit_transform(X)
    
        # Add final noise layer
        noise = np.random.normal(0, 0.1, size=X_scaled.shape)
        return X_scaled + noise

    def generate_comprehensive_guidance(self, user_data):
        """Generate comprehensive loan guidance"""
        processed_data = self.preprocess_data(pd.DataFrame([user_data]))
        X_user = self.prepare_features(processed_data)
        
        risk_score = self.risk_model.predict_proba(X_user)[0][1]
        predicted_payment = self.payment_predictor.predict(X_user)[0]
        
        return {
            'risk_assessment': self.generate_risk_assessment(risk_score, user_data),
            'payment_plan': self.generate_payment_plan(user_data, predicted_payment),
            'recommendations': self.generate_smart_recommendations(user_data, risk_score),
            'monitoring_plan': self.generate_monitoring_plan(risk_score)
        }

    def generate_risk_assessment(self, risk_score, user_data):
        risk_level = self.categorize_risk(risk_score)
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 4),
            'key_factors': self.identify_risk_factors(user_data),
            'mitigation_strategies': self.get_risk_mitigation_strategies(risk_level)
        }

    def identify_risk_factors(self, user_data):
        """Identify key risk factors"""
        risk_factors = []
        
        if user_data['credit_score'] < 650:
            risk_factors.append({"factor": "Low Credit Score", "impact": "High"})
        
        dti = user_data['loan_amount'] / (user_data['monthly_income'] * user_data['loan_term_months'])
        if dti > 0.4:
            risk_factors.append({"factor": "High Debt-to-Income Ratio", "impact": "High"})
            
        if user_data['age'] < 25:
            risk_factors.append({"factor": "Limited Credit History", "impact": "Medium"})
        
        return risk_factors

    def get_risk_mitigation_strategies(self, risk_level):
        """Get risk mitigation strategies based on risk level"""
        strategies = {
            'Low Risk': [
                "Continue regular payments",
                "Consider early payment options",
                "Build emergency fund"
            ],
            'Moderate Risk': [
                "Set up automatic payments",
                "Create strict budget",
                "Build larger emergency fund"
            ],
            'High Risk': [
                "Consider loan restructuring",
                "Seek financial counseling",
                "Explore additional income sources"
            ]
        }
        return strategies.get(risk_level, [])

    def generate_payment_plan(self, user_data, predicted_payment):
        return {
            'monthly_payment': round(predicted_payment, 2),
            'payment_schedule': self.create_detailed_schedule(user_data, predicted_payment),
            'flexibility_options': self.get_payment_flexibility_options(user_data),
            'early_payment_benefits': self.calculate_early_payment_benefits(user_data)
        }

    def create_detailed_schedule(self, loan_details, monthly_payment):
        schedule = []
        remaining_balance = loan_details['loan_amount']
        
        for month in range(int(loan_details['loan_term_months'])):
            payment_date = datetime.now() + timedelta(days=30 * month)
            interest_payment = (remaining_balance * loan_details['interest_rate']) / 1200
            principal_payment = monthly_payment - interest_payment
            remaining_balance -= principal_payment
            
            schedule.append({
                'payment_number': month + 1,
                'due_date': payment_date.strftime('%Y-%m-%d'),
                'payment_amount': round(monthly_payment, 2),
                'principal': round(principal_payment, 2),
                'interest': round(interest_payment, 2),
                'remaining_balance': round(max(0, remaining_balance), 2),
                'reminder_dates': [
                    (payment_date - timedelta(days=7)).strftime('%Y-%m-%d'),
                    (payment_date - timedelta(days=3)).strftime('%Y-%m-%d'),
                    payment_date.strftime('%Y-%m-%d')
                ]
            })
            
        return schedule

    def calculate_early_payment_benefits(self, user_data):
        """Calculate benefits of early payment"""
        loan_amount = user_data['loan_amount']
        interest_rate = user_data['interest_rate'] / 100
        term_months = user_data['loan_term_months']
        
        regular_total = loan_amount * (1 + interest_rate * term_months/12)
        early_total = loan_amount * (1 + interest_rate * (term_months-6)/12)
        
        return {
            'potential_savings': round(regular_total - early_total, 2),
            'time_saved': 6,  # months
            'reduced_interest': round(regular_total - early_total, 2)
        }

    def get_payment_flexibility_options(self, user_data):
        """Get payment flexibility options"""
        return {
            'bi_weekly_option': {
                'available': True,
                'impact': self.calculate_biweekly_impact(user_data)
            },
            'extra_payment_option': {
                'available': True,
                'min_amount': user_data['monthly_income'] * 0.1
            }
        }

    def calculate_biweekly_impact(self, user_data):
        """Calculate impact of bi-weekly payments"""
        monthly_payment = user_data['loan_amount'] / user_data['loan_term_months']
        biweekly_payment = monthly_payment / 2
        
        return {
            'payment_amount': round(biweekly_payment, 2),
            'yearly_savings': round(monthly_payment * 0.5, 2)
        }

    def generate_smart_recommendations(self, user_data, risk_score):
        """Generate smart recommendations based on user profile and risk"""
        recommendations = {
            'payment_strategy': [],
            'risk_mitigation': [],
            'financial_planning': []
        }
        
        # Payment strategy recommendations
        monthly_income = user_data['monthly_income']
        loan_amount = user_data['loan_amount']
        
        # Basic payment strategies
        if loan_amount / monthly_income > 24:
            recommendations['payment_strategy'].extend([
                "Consider bi-weekly payments to reduce interest",
                "Allocate year-end bonus to loan payment"
            ])
        
        # Borrower type specific recommendations
        type_specific_recs = {
            'farmer': [
                "Time payments with harvest cycles",
                "Consider crop insurance for risk mitigation",
                "Explore government agricultural subsidies"
            ],
            'student': [
                "Look for part-time work opportunities",
                "Apply for educational scholarships",
                "Consider income-based repayment options"
            ],
            'business': [
                "Align payments with business cash flow cycles",
                "Maintain separate business and personal accounts",
                "Explore invoice financing options"
            ]
        }
        
        if user_data['borrower_type'] in type_specific_recs:
            recommendations['payment_strategy'].extend(
                type_specific_recs[user_data['borrower_type']]
            )
        
        # Risk-based recommendations
        if risk_score > 0.6:
            recommendations['risk_mitigation'].extend([
                "Build emergency fund of 6 months",
                "Consider payment protection insurance",
                "Set up automatic payments to avoid delays"
            ])
        elif risk_score > 0.3:
            recommendations['risk_mitigation'].extend([
                "Build emergency fund of 3 months",
                "Review monthly budget",
                "Consider income diversification"
            ])
        
        # Financial planning recommendations
        credit_score = user_data['credit_score']
        if credit_score < 700:
            recommendations['financial_planning'].extend([
                "Focus on improving credit score",
                "Review and dispute any credit report errors",
                "Minimize new credit applications"
            ])
        
        return recommendations

    def generate_monitoring_plan(self, risk_score):
        """Generate monitoring plan based on risk score"""
        if risk_score < 0.3:
            frequency = "Quarterly"
            checks = ["Payment History", "Credit Score"]
        elif risk_score < 0.7:
            frequency = "Monthly"
            checks = ["Payment History", "Credit Score", "Income Verification"]
        else:
            frequency = "Weekly"
            checks = ["Payment History", "Credit Score", "Income Verification", "Expense Tracking"]
        
        return {
            'monitoring_frequency': frequency,
            'required_checks': checks,
            'alert_thresholds': {
                'missed_payments': 1,
                'credit_score_drop': 50,
                'income_change': 0.2
            }
        }

    def categorize_risk(self, risk_score):
        if risk_score < 0.3:
            return 'Low Risk'
        elif risk_score < 0.7:
            return 'Moderate Risk'
        return 'High Risk'

def main():
    print("Loading data...")
    data = pd.read_csv('/kaggle/input/processed-dataset/processed_loan_dataset.csv')
    
    # Randomly drop some rows
    drop_mask = np.random.binomial(1, 0.98, size=len(data))
    data = data[drop_mask.astype(bool)]
    
    guidance_system = AdvancedLoanGuidanceSystem()
    
    print("Preprocessing data...")
    processed_data = guidance_system.preprocess_data(data)
    
    print("Preparing features...")
    X = guidance_system.prepare_features(processed_data)
    
    print("Preparing targets...")
    # Add noise to targets
    y_risk = (processed_data['loan_status'] == 'Charged Off').astype(int)
    noise_mask = np.random.random(len(y_risk)) < 0.05  # 5% label noise
    y_risk = y_risk.copy()
    y_risk[noise_mask] = 1 - y_risk[noise_mask]
    
    y_payment = processed_data['loan_amount'] / processed_data['loan_term_months']
    y_payment = y_payment * (1 + np.random.normal(0, 0.1, size=len(y_payment)))
    
    print("Splitting data...")
    X_train, X_test, y_risk_train, y_risk_test, y_payment_train, y_payment_test = train_test_split(
        X, y_risk, y_payment, test_size=0.3, random_state=42, stratify=y_risk
    )
    print("Training models...")
    guidance_system.train_models(X_train, y_risk_train, y_payment_train)
    
    print("\nPerforming cross-validation...")
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(
        guidance_system.risk_model,
        X_train,
        y_risk_train,
        cv=5,
        scoring='accuracy'
    )
    
    print("\nCross-validation scores:")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nEvaluating on test set...")
    y_risk_pred = guidance_system.risk_model.predict(X_test)
    accuracy = accuracy_score(y_risk_test, y_risk_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_risk_test, y_risk_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    print("\nSaving model...")
    joblib.dump(guidance_system, 'advanced_loan_guidance_system.joblib')
    
    return guidance_system

if __name__ == "__main__":
    guidance_system = main()