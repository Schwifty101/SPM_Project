"""
Loyalty AI Agent - Core Logic
Implements customer segmentation, reward optimization, and churn prediction
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class LoyaltyAgent:
    """
    AI Agent for customer loyalty optimization
    Implements RFM analysis, segmentation, and personalized reward recommendations
    """

    # Reward types with costs (in PKR)
    REWARD_CATALOG = {
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

    # Churn risk thresholds
    CHURN_THRESHOLDS = {
        "high_risk": 0.7,      # >70% churn probability
        "medium_risk": 0.4,    # 40-70% churn probability
        "low_risk": 0.0        # <40% churn probability
    }

    def __init__(self, customers_file: str = "data/customers.json",
                 transactions_file: str = "data/transactions.json"):
        """
        Initialize Loyalty Agent

        Args:
            customers_file: Path to customers JSON file
            transactions_file: Path to transactions JSON file
        """
        self.customers_file = customers_file
        self.transactions_file = transactions_file
        self.customers = []
        self.transactions = []
        self.rfm_scores = {}

        # Load data if files exist
        self._load_data()

    def _load_data(self):
        """Load customer and transaction data from JSON files"""
        try:
            customers_path = Path(self.customers_file)
            if customers_path.exists():
                with open(customers_path, 'r') as f:
                    self.customers = json.load(f)
                print(f"✓ Loaded {len(self.customers)} customers")

            transactions_path = Path(self.transactions_file)
            if transactions_path.exists():
                with open(transactions_path, 'r') as f:
                    self.transactions = json.load(f)
                print(f"✓ Loaded {len(self.transactions)} transactions")

        except Exception as e:
            print(f"Warning: Could not load data files: {e}")

    def calculate_rfm_score(self, customer_id: str) -> Dict[str, float]:
        """
        Calculate RFM (Recency, Frequency, Monetary) score for a customer

        Args:
            customer_id: Unique customer identifier

        Returns:
            Dictionary with R, F, M scores and normalized RFM score (0-100)
        """
        # Get customer data
        customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
        if not customer:
            return {"recency": 0, "frequency": 0, "monetary": 0, "rfm_score": 0, "error": "Customer not found"}

        # Get customer transactions
        customer_txns = [t for t in self.transactions
                        if t['customer_id'] == customer_id and t['status'] == 'Completed']

        if not customer_txns:
            return {"recency": 0, "frequency": 0, "monetary": 0, "rfm_score": 0}

        # Calculate Recency (days since last purchase - lower is better)
        last_purchase = datetime.strptime(customer['last_purchase_date'], "%Y-%m-%d")
        today = datetime.now()
        recency_days = (today - last_purchase).days

        # Normalize recency (0-100, where 100 is most recent)
        # Using inverse exponential: score decreases as days increase
        recency_score = max(0, 100 - (recency_days / 3.65))  # Decay over ~1 year

        # Calculate Frequency (number of purchases)
        frequency = customer['total_purchases']

        # Normalize frequency (0-100)
        # Assuming max frequency of 150 purchases (from data generation)
        frequency_score = min(100, (frequency / 150) * 100)

        # Calculate Monetary (total lifetime value)
        monetary = customer['lifetime_value']

        # Normalize monetary (0-100)
        # Assuming max LTV of 300,000 PKR
        monetary_score = min(100, (monetary / 300000) * 100)

        # Combined RFM score (weighted average)
        rfm_score = (recency_score * 0.3 + frequency_score * 0.35 + monetary_score * 0.35)

        return {
            "recency": round(recency_score, 2),
            "frequency": round(frequency_score, 2),
            "monetary": round(monetary_score, 2),
            "rfm_score": round(rfm_score, 2),
            "recency_days": recency_days,
            "total_purchases": frequency,
            "lifetime_value": round(monetary, 2)
        }

    def predict_churn_probability(self, customer_id: str) -> float:
        """
        Predict churn probability for a customer (0-1 scale)

        Uses multiple factors:
        - Recency of last purchase
        - Purchase frequency decline
        - Engagement score
        - RFM score

        Args:
            customer_id: Unique customer identifier

        Returns:
            Churn probability (0-1, where 1 = high risk)
        """
        customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
        if not customer:
            return 1.0  # Unknown customer = high risk

        # Factor 1: Recency risk (days since last purchase)
        last_purchase = datetime.strptime(customer['last_purchase_date'], "%Y-%m-%d")
        days_since_purchase = (datetime.now() - last_purchase).days

        # Recency risk increases exponentially after 90 days
        if days_since_purchase < 30:
            recency_risk = 0.1
        elif days_since_purchase < 90:
            recency_risk = 0.3
        elif days_since_purchase < 180:
            recency_risk = 0.6
        else:
            recency_risk = 0.9

        # Factor 2: Frequency risk (purchase frequency)
        frequency = customer['purchase_frequency']
        if frequency > 2:  # >2 purchases/month
            frequency_risk = 0.1
        elif frequency > 1:
            frequency_risk = 0.3
        elif frequency > 0.5:
            frequency_risk = 0.5
        else:
            frequency_risk = 0.8

        # Factor 3: Engagement risk (inverse of engagement score)
        engagement = customer['engagement_score']
        engagement_risk = 1 - (engagement / 100)

        # Factor 4: RFM risk
        rfm = self.calculate_rfm_score(customer_id)
        rfm_risk = 1 - (rfm['rfm_score'] / 100)

        # Weighted churn probability
        churn_probability = (
            recency_risk * 0.35 +
            frequency_risk * 0.25 +
            engagement_risk * 0.25 +
            rfm_risk * 0.15
        )

        return round(min(1.0, churn_probability), 3)

    def segment_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Perform advanced customer segmentation

        Args:
            customer_id: Unique customer identifier

        Returns:
            Dictionary with segment, tier, and behavioral insights
        """
        customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
        if not customer:
            return {"error": "Customer not found"}

        rfm = self.calculate_rfm_score(customer_id)
        churn_prob = self.predict_churn_probability(customer_id)

        # Determine detailed segment
        rfm_score = rfm['rfm_score']

        if rfm_score >= 75:
            if churn_prob < 0.3:
                detailed_segment = "Champion"
            else:
                detailed_segment = "At-Risk Champion"
        elif rfm_score >= 50:
            if churn_prob < 0.4:
                detailed_segment = "Loyal Customer"
            else:
                detailed_segment = "At-Risk Loyal"
        elif rfm_score >= 30:
            if churn_prob < 0.5:
                detailed_segment = "Potential Loyalist"
            else:
                detailed_segment = "Hibernating"
        else:
            if customer['total_purchases'] < 3:
                detailed_segment = "New Customer"
            else:
                detailed_segment = "Lost Customer"

        return {
            "customer_id": customer_id,
            "basic_segment": customer['segment'],
            "loyalty_tier": customer['loyalty_tier'],
            "detailed_segment": detailed_segment,
            "rfm_score": rfm['rfm_score'],
            "churn_probability": churn_prob,
            "is_at_risk": churn_prob >= self.CHURN_THRESHOLDS['medium_risk'],
            "engagement_level": "High" if customer['engagement_score'] >= 70 else
                               "Medium" if customer['engagement_score'] >= 40 else "Low"
        }

    def recommend_reward(self, customer_id: str) -> Dict[str, Any]:
        """
        Recommend personalized reward/incentive for a customer

        Uses multi-armed bandit-inspired approach with contextual factors:
        - Customer segment and tier
        - Churn risk
        - RFM score
        - Purchase history

        Args:
            customer_id: Unique customer identifier

        Returns:
            Dictionary with reward recommendation and reasoning
        """
        customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
        if not customer:
            return {"error": "Customer not found"}

        # Get customer insights
        segmentation = self.segment_customer(customer_id)
        churn_prob = segmentation['churn_probability']
        rfm_score = segmentation['rfm_score']
        detailed_segment = segmentation['detailed_segment']

        # Reward selection logic based on customer profile
        recommended_rewards = []

        # High-value at-risk customers (prevent churn)
        if detailed_segment in ["At-Risk Champion", "At-Risk Loyal"] or churn_prob >= 0.7:
            recommended_rewards = [
                ("vip_upgrade", 0.9),
                ("premium_discount", 0.85),
                ("gift_voucher", 0.8),
                ("cashback", 0.75)
            ]
            strategy = "Churn Prevention - High Value Retention"

        # Champions and Loyal Customers (maintain engagement)
        elif detailed_segment in ["Champion", "Loyal Customer"]:
            recommended_rewards = [
                ("early_access", 0.9),
                ("vip_upgrade", 0.8),
                ("birthday_special", 0.75),
                ("premium_discount", 0.7)
            ]
            strategy = "Engagement & Loyalty Reinforcement"

        # Potential Loyalists (nurture growth)
        elif detailed_segment == "Potential Loyalist":
            recommended_rewards = [
                ("loyalty_points", 0.9),
                ("bundle_offer", 0.85),
                ("standard_discount", 0.8),
                ("free_shipping", 0.7)
            ]
            strategy = "Growth & Upsell"

        # New Customers (encourage repeat purchase)
        elif detailed_segment == "New Customer":
            recommended_rewards = [
                ("standard_discount", 0.9),
                ("free_shipping", 0.85),
                ("loyalty_points", 0.8)
            ]
            strategy = "New Customer Activation"

        # Hibernating/Lost Customers (win-back)
        else:
            recommended_rewards = [
                ("premium_discount", 0.9),
                ("gift_voucher", 0.85),
                ("cashback", 0.8)
            ]
            strategy = "Win-Back Campaign"

        # Select top reward
        if recommended_rewards:
            top_reward_key, confidence = recommended_rewards[0]
            top_reward = self.REWARD_CATALOG[top_reward_key]
        else:
            top_reward_key = "standard_discount"
            top_reward = self.REWARD_CATALOG[top_reward_key]
            confidence = 0.5
            strategy = "Default Reward"

        # Calculate expected ROI
        expected_retention_lift = confidence * 0.3  # 30% max retention improvement
        customer_ltv = customer['lifetime_value']
        expected_value = customer_ltv * expected_retention_lift
        reward_cost = top_reward['cost']
        expected_roi = ((expected_value - reward_cost) / reward_cost) * 100 if reward_cost > 0 else 0

        return {
            "customer_id": customer_id,
            "recommended_reward": top_reward_key,
            "reward_details": top_reward,
            "confidence": round(confidence, 2),
            "strategy": strategy,
            "alternative_rewards": [
                {"reward": self.REWARD_CATALOG[r[0]]['name'], "confidence": r[1]}
                for r in recommended_rewards[1:3]
            ] if len(recommended_rewards) > 1 else [],
            "expected_retention_lift": f"{expected_retention_lift*100:.1f}%",
            "expected_roi": f"{expected_roi:.1f}%",
            "reasoning": {
                "segment": detailed_segment,
                "churn_risk": "High" if churn_prob >= 0.7 else "Medium" if churn_prob >= 0.4 else "Low",
                "rfm_score": rfm_score,
                "lifetime_value": customer_ltv
            }
        }

    def analyze_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Comprehensive customer analysis combining all insights

        Args:
            customer_id: Unique customer identifier

        Returns:
            Complete analysis with segmentation, churn risk, and recommendations
        """
        customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
        if not customer:
            return {"error": "Customer not found", "customer_id": customer_id}

        # Gather all insights
        rfm = self.calculate_rfm_score(customer_id)
        segmentation = self.segment_customer(customer_id)
        churn_prediction = self.predict_churn_probability(customer_id)
        reward_recommendation = self.recommend_reward(customer_id)

        return {
            "customer_id": customer_id,
            "profile": {
                "segment": customer['segment'],
                "loyalty_tier": customer['loyalty_tier'],
                "registration_date": customer['registration_date'],
                "last_purchase_date": customer['last_purchase_date'],
                "is_active": customer['is_active']
            },
            "rfm_analysis": rfm,
            "segmentation": segmentation,
            "churn_prediction": {
                "probability": churn_prediction,
                "risk_level": "High" if churn_prediction >= 0.7 else "Medium" if churn_prediction >= 0.4 else "Low",
                "predicted_retention": f"{(1-churn_prediction)*100:.1f}%"
            },
            "recommendation": reward_recommendation,
            "kpis": {
                "lifetime_value": customer['lifetime_value'],
                "total_purchases": customer['total_purchases'],
                "avg_order_value": customer['avg_order_value'],
                "engagement_score": customer['engagement_score']
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def batch_analyze(self, customer_ids: List[str] = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple customers in batch

        Args:
            customer_ids: Optional list of specific customer IDs
            limit: Maximum number of customers to analyze

        Returns:
            List of customer analyses
        """
        if customer_ids:
            ids_to_analyze = customer_ids
        else:
            ids_to_analyze = [c['customer_id'] for c in self.customers]

        if limit:
            ids_to_analyze = ids_to_analyze[:limit]

        results = []
        for customer_id in ids_to_analyze:
            analysis = self.analyze_customer(customer_id)
            results.append(analysis)

        return results

    def get_high_value_at_risk_customers(self, threshold: float = 0.6, min_ltv: float = 50000) -> List[Dict]:
        """
        Identify high-value customers at risk of churning

        Args:
            threshold: Churn probability threshold
            min_ltv: Minimum lifetime value to consider

        Returns:
            List of at-risk high-value customers with recommendations
        """
        at_risk_customers = []

        for customer in self.customers:
            if customer['lifetime_value'] >= min_ltv:
                customer_id = customer['customer_id']
                churn_prob = self.predict_churn_probability(customer_id)

                if churn_prob >= threshold:
                    analysis = self.analyze_customer(customer_id)
                    at_risk_customers.append(analysis)

        # Sort by lifetime value descending
        at_risk_customers.sort(key=lambda x: x['kpis']['lifetime_value'], reverse=True)

        return at_risk_customers


def main():
    """Demo: Analyze sample customers"""
    print("="*60)
    print("LOYALTY AI AGENT - DEMO")
    print("="*60)

    # Initialize agent
    agent = LoyaltyAgent()

    if not agent.customers:
        print("\n⚠ No customer data found. Please run data_generator.py first.")
        return

    # Analyze first 5 customers
    print(f"\nAnalyzing sample customers...\n")

    sample_customers = agent.customers[:5]
    for customer in sample_customers:
        customer_id = customer['customer_id']
        analysis = agent.analyze_customer(customer_id)

        print(f"\n{'─'*60}")
        print(f"Customer: {customer_id} | Segment: {analysis['segmentation']['detailed_segment']}")
        print(f"{'─'*60}")
        print(f"RFM Score: {analysis['rfm_analysis']['rfm_score']}/100")
        print(f"Churn Risk: {analysis['churn_prediction']['risk_level']} ({analysis['churn_prediction']['probability']})")
        print(f"Lifetime Value: PKR {analysis['kpis']['lifetime_value']:,.2f}")
        print(f"\nRecommended Action: {analysis['recommendation']['strategy']}")
        print(f"Reward: {analysis['recommendation']['reward_details']['name']}")
        print(f"Expected ROI: {analysis['recommendation']['expected_roi']}")

    # High-value at-risk analysis
    print(f"\n\n{'='*60}")
    print("HIGH-VALUE AT-RISK CUSTOMERS")
    print(f"{'='*60}")

    at_risk = agent.get_high_value_at_risk_customers(threshold=0.6, min_ltv=50000)
    print(f"\nFound {len(at_risk)} high-value customers at risk of churning\n")

    for i, customer_analysis in enumerate(at_risk[:3], 1):
        print(f"{i}. {customer_analysis['customer_id']}")
        print(f"   LTV: PKR {customer_analysis['kpis']['lifetime_value']:,.2f}")
        print(f"   Churn Risk: {customer_analysis['churn_prediction']['probability']}")
        print(f"   Action: {customer_analysis['recommendation']['reward_details']['name']}\n")

    print("="*60)


if __name__ == "__main__":
    main()
