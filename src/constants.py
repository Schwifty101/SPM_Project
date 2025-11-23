"""
Constants and configuration for Loyalty AI Agent
"""

from typing import Dict, Any

# RFM Scoring Parameters
RFM_RECENCY_DECAY_DAYS = 3.65  # Decay factor for recency (365 days)
RFM_MAX_FREQUENCY = 150  # Maximum expected purchase frequency
RFM_MAX_LTV = 300000  # Maximum expected lifetime value (PKR)

# RFM Weight Distribution
RFM_WEIGHT_RECENCY = 0.30
RFM_WEIGHT_FREQUENCY = 0.35
RFM_WEIGHT_MONETARY = 0.35

# Churn Prediction Thresholds (days)
CHURN_RECENCY_RECENT = 30
CHURN_RECENCY_MODERATE = 90
CHURN_RECENCY_OLD = 180

# Churn Risk Levels
CHURN_RISK_HIGH = 0.7
CHURN_RISK_MEDIUM = 0.4
CHURN_RISK_LOW = 0.0

# Churn Prediction Weights
CHURN_WEIGHT_RECENCY = 0.35
CHURN_WEIGHT_FREQUENCY = 0.25
CHURN_WEIGHT_ENGAGEMENT = 0.25
CHURN_WEIGHT_RFM = 0.15

# Purchase Frequency Thresholds (purchases per month)
FREQUENCY_HIGH = 2.0
FREQUENCY_MEDIUM = 1.0
FREQUENCY_LOW = 0.5

# Engagement Score Thresholds
ENGAGEMENT_HIGH = 70
ENGAGEMENT_MEDIUM = 40

# RFM Segmentation Thresholds
RFM_CHAMPION_THRESHOLD = 75
RFM_LOYAL_THRESHOLD = 50
RFM_POTENTIAL_THRESHOLD = 30
NEW_CUSTOMER_PURCHASE_LIMIT = 3

# Reward Catalog with costs (PKR)
REWARD_CATALOG: Dict[str, Dict[str, Any]] = {
    "premium_discount": {"name": "20% Premium Discount", "cost": 500, "value": "20% off next purchase"},
    "standard_discount": {"name": "10% Standard Discount", "cost": 200, "value": "10% off next purchase"},
    "free_shipping": {"name": "Free Shipping Voucher", "cost": 150, "value": "Free shipping on next order"},
    "gift_voucher": {"name": "PKR 500 Gift Voucher", "cost": 500, "value": "PKR 500 gift card"},
    "loyalty_points": {"name": "1000 Loyalty Points", "cost": 100, "value": "1000 bonus points"},
    "early_access": {"name": "Early Access to Sales", "cost": 50, "value": "24-hour early access"},
    "birthday_special": {"name": "Birthday Special Offer", "cost": 300, "value": "Special birthday gift"},
    "bundle_offer": {"name": "Bundle Deal", "cost": 250, "value": "Buy 2 Get 1 Free"},
    "vip_upgrade": {"name": "VIP Tier Upgrade", "cost": 1000, "value": "Upgrade to next tier"},
    "cashback": {"name": "15% Cashback", "cost": 350, "value": "15% cashback on purchase"}
}

# Reward Strategy Parameters
MAX_RETENTION_LIFT = 0.30  # Maximum 30% retention improvement

# High-Value Customer Thresholds
HIGH_VALUE_CHURN_THRESHOLD = 0.6
HIGH_VALUE_MIN_LTV = 50000  # PKR

# Default file paths
DEFAULT_CUSTOMERS_FILE = "data/customers.json"
DEFAULT_TRANSACTIONS_FILE = "data/transactions.json"
