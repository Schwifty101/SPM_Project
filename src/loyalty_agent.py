"""
Loyalty AI Agent - Core Logic
Implements customer segmentation, reward optimization, and churn prediction
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from src.constants import (
        REWARD_CATALOG, RFM_RECENCY_DECAY_DAYS, RFM_MAX_FREQUENCY, RFM_MAX_LTV,
        RFM_WEIGHT_RECENCY, RFM_WEIGHT_FREQUENCY, RFM_WEIGHT_MONETARY,
        CHURN_RECENCY_RECENT, CHURN_RECENCY_MODERATE, CHURN_RECENCY_OLD,
        CHURN_RISK_HIGH, CHURN_RISK_MEDIUM, CHURN_WEIGHT_RECENCY, CHURN_WEIGHT_FREQUENCY,
        CHURN_WEIGHT_ENGAGEMENT, CHURN_WEIGHT_RFM, FREQUENCY_HIGH, FREQUENCY_MEDIUM,
        FREQUENCY_LOW, ENGAGEMENT_HIGH, ENGAGEMENT_MEDIUM, RFM_CHAMPION_THRESHOLD,
        RFM_LOYAL_THRESHOLD, RFM_POTENTIAL_THRESHOLD, NEW_CUSTOMER_PURCHASE_LIMIT,
        MAX_RETENTION_LIFT, HIGH_VALUE_CHURN_THRESHOLD, HIGH_VALUE_MIN_LTV,
        DEFAULT_CUSTOMERS_FILE, DEFAULT_TRANSACTIONS_FILE
    )
    from src.validators import validate_customer_id, validate_probability, validate_limit, CustomerNotFoundError
    from src.logger import get_logger
except ImportError:
    from constants import (
        REWARD_CATALOG, RFM_RECENCY_DECAY_DAYS, RFM_MAX_FREQUENCY, RFM_MAX_LTV,
        RFM_WEIGHT_RECENCY, RFM_WEIGHT_FREQUENCY, RFM_WEIGHT_MONETARY,
        CHURN_RECENCY_RECENT, CHURN_RECENCY_MODERATE, CHURN_RECENCY_OLD,
        CHURN_RISK_HIGH, CHURN_RISK_MEDIUM, CHURN_WEIGHT_RECENCY, CHURN_WEIGHT_FREQUENCY,
        CHURN_WEIGHT_ENGAGEMENT, CHURN_WEIGHT_RFM, FREQUENCY_HIGH, FREQUENCY_MEDIUM,
        FREQUENCY_LOW, ENGAGEMENT_HIGH, ENGAGEMENT_MEDIUM, RFM_CHAMPION_THRESHOLD,
        RFM_LOYAL_THRESHOLD, RFM_POTENTIAL_THRESHOLD, NEW_CUSTOMER_PURCHASE_LIMIT,
        MAX_RETENTION_LIFT, HIGH_VALUE_CHURN_THRESHOLD, HIGH_VALUE_MIN_LTV,
        DEFAULT_CUSTOMERS_FILE, DEFAULT_TRANSACTIONS_FILE
    )
    from validators import validate_customer_id, validate_probability, validate_limit, CustomerNotFoundError
    from logger import get_logger


class LoyaltyAgent:
    """AI Agent for customer loyalty optimization with RFM analysis and personalized recommendations"""

    def __init__(self, customers_file: str = DEFAULT_CUSTOMERS_FILE,
                 transactions_file: str = DEFAULT_TRANSACTIONS_FILE):
        """Initialize Loyalty Agent and load data"""
        self.logger = get_logger(__name__)
        self.customers_file = customers_file
        self.transactions_file = transactions_file
        self.customers: List[Dict] = []
        self.transactions: List[Dict] = []
        # O(1) lookup indexes
        self.customer_index: Dict[str, Dict] = {}
        self.transaction_index: Dict[str, List[Dict]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load customer and transaction data from JSON files and build indexes"""
        try:
            customers_path = Path(self.customers_file)
            if customers_path.exists():
                with open(customers_path, 'r') as f:
                    self.customers = json.load(f)
                self.logger.info(f"Loaded {len(self.customers)} customers")

            transactions_path = Path(self.transactions_file)
            if transactions_path.exists():
                with open(transactions_path, 'r') as f:
                    self.transactions = json.load(f)
                self.logger.info(f"Loaded {len(self.transactions)} transactions")

            # Build indexes for O(1) lookups
            self._build_indexes()
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _build_indexes(self) -> None:
        """Build customer and transaction indexes for O(1) lookups"""
        # Build customer index
        self.customer_index = {c['customer_id']: c for c in self.customers}

        # Build transaction index (group by customer_id)
        self.transaction_index = {}
        for txn in self.transactions:
            customer_id = txn['customer_id']
            if customer_id not in self.transaction_index:
                self.transaction_index[customer_id] = []
            self.transaction_index[customer_id].append(txn)

    def _get_customer(self, customer_id: str) -> Optional[Dict]:
        """Get customer by ID using O(1) index lookup"""
        return self.customer_index.get(customer_id)

    def _get_customer_transactions(self, customer_id: str) -> List[Dict]:
        """Get completed transactions for a customer using O(1) index lookup"""
        txns = self.transaction_index.get(customer_id, [])
        return [t for t in txns if t['status'] == 'Completed']

    def calculate_rfm_score(self, customer_id: str) -> Dict[str, float]:
        """Calculate RFM (Recency, Frequency, Monetary) score"""
        customer_id = validate_customer_id(customer_id)
        customer = self._get_customer(customer_id)

        if not customer:
            self.logger.warning(f"Customer not found: {customer_id}")
            raise CustomerNotFoundError(f"Customer not found: {customer_id}")

        customer_txns = self._get_customer_transactions(customer_id)
        if not customer_txns:
            return {"recency": 0, "frequency": 0, "monetary": 0, "rfm_score": 0}

        # Recency: days since last purchase (lower is better)
        last_purchase = datetime.strptime(customer['last_purchase_date'], "%Y-%m-%d")
        recency_days = (datetime.now() - last_purchase).days
        recency_score = max(0, 100 - (recency_days / RFM_RECENCY_DECAY_DAYS))

        # Frequency: number of purchases
        frequency = customer['total_purchases']
        frequency_score = min(100, (frequency / RFM_MAX_FREQUENCY) * 100)

        # Monetary: total lifetime value
        monetary = customer['lifetime_value']
        monetary_score = min(100, (monetary / RFM_MAX_LTV) * 100)

        # Combined RFM score (weighted average)
        rfm_score = (recency_score * RFM_WEIGHT_RECENCY +
                    frequency_score * RFM_WEIGHT_FREQUENCY +
                    monetary_score * RFM_WEIGHT_MONETARY)

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
        """Predict churn probability using multi-factor analysis"""
        customer_id = validate_customer_id(customer_id)
        customer = self._get_customer(customer_id)

        if not customer:
            self.logger.warning(f"Customer not found: {customer_id}")
            raise CustomerNotFoundError(f"Customer not found: {customer_id}")

        # Recency risk
        last_purchase = datetime.strptime(customer['last_purchase_date'], "%Y-%m-%d")
        days_since_purchase = (datetime.now() - last_purchase).days

        if days_since_purchase < CHURN_RECENCY_RECENT:
            recency_risk = 0.1
        elif days_since_purchase < CHURN_RECENCY_MODERATE:
            recency_risk = 0.3
        elif days_since_purchase < CHURN_RECENCY_OLD:
            recency_risk = 0.6
        else:
            recency_risk = 0.9

        # Frequency risk
        frequency = customer['purchase_frequency']
        if frequency > FREQUENCY_HIGH:
            frequency_risk = 0.1
        elif frequency > FREQUENCY_MEDIUM:
            frequency_risk = 0.3
        elif frequency > FREQUENCY_LOW:
            frequency_risk = 0.5
        else:
            frequency_risk = 0.8

        # Engagement risk
        engagement_risk = 1 - (customer['engagement_score'] / 100)

        # RFM risk
        rfm = self.calculate_rfm_score(customer_id)
        rfm_risk = 1 - (rfm['rfm_score'] / 100)

        # Weighted churn probability
        churn_probability = (
            recency_risk * CHURN_WEIGHT_RECENCY +
            frequency_risk * CHURN_WEIGHT_FREQUENCY +
            engagement_risk * CHURN_WEIGHT_ENGAGEMENT +
            rfm_risk * CHURN_WEIGHT_RFM
        )

        return round(min(1.0, churn_probability), 3)

    def segment_customer(self, customer_id: str) -> Dict[str, Any]:
        """Perform advanced customer segmentation"""
        customer_id = validate_customer_id(customer_id)
        customer = self._get_customer(customer_id)

        if not customer:
            raise CustomerNotFoundError(f"Customer not found: {customer_id}")

        rfm = self.calculate_rfm_score(customer_id)
        churn_prob = self.predict_churn_probability(customer_id)
        rfm_score = rfm['rfm_score']

        # Determine detailed segment
        if rfm_score >= RFM_CHAMPION_THRESHOLD:
            detailed_segment = "Champion" if churn_prob < 0.3 else "At-Risk Champion"
        elif rfm_score >= RFM_LOYAL_THRESHOLD:
            detailed_segment = "Loyal Customer" if churn_prob < 0.4 else "At-Risk Loyal"
        elif rfm_score >= RFM_POTENTIAL_THRESHOLD:
            detailed_segment = "Potential Loyalist" if churn_prob < 0.5 else "Hibernating"
        else:
            detailed_segment = ("New Customer" if customer['total_purchases'] < NEW_CUSTOMER_PURCHASE_LIMIT
                              else "Lost Customer")

        return {
            "customer_id": customer_id,
            "basic_segment": customer['segment'],
            "loyalty_tier": customer['loyalty_tier'],
            "detailed_segment": detailed_segment,
            "rfm_score": rfm['rfm_score'],
            "churn_probability": churn_prob,
            "is_at_risk": churn_prob >= CHURN_RISK_MEDIUM,
            "engagement_level": ("High" if customer['engagement_score'] >= ENGAGEMENT_HIGH else
                               "Medium" if customer['engagement_score'] >= ENGAGEMENT_MEDIUM else "Low")
        }

    def _select_reward_strategy(self, detailed_segment: str, churn_prob: float) -> Tuple[List[Tuple[str, float]], str]:
        """Select reward strategy based on customer segment and churn risk"""

        if detailed_segment in ["At-Risk Champion", "At-Risk Loyal"] or churn_prob >= CHURN_RISK_HIGH:
            return [("vip_upgrade", 0.9), ("premium_discount", 0.85),
                   ("gift_voucher", 0.8), ("cashback", 0.75)], "Churn Prevention - High Value Retention"

        if detailed_segment in ["Champion", "Loyal Customer"]:
            return [("early_access", 0.9), ("vip_upgrade", 0.8),
                   ("birthday_special", 0.75), ("premium_discount", 0.7)], "Engagement & Loyalty Reinforcement"

        if detailed_segment == "Potential Loyalist":
            return [("loyalty_points", 0.9), ("bundle_offer", 0.85),
                   ("standard_discount", 0.8), ("free_shipping", 0.7)], "Growth & Upsell"

        if detailed_segment == "New Customer":
            return [("standard_discount", 0.9), ("free_shipping", 0.85),
                   ("loyalty_points", 0.8)], "New Customer Activation"

        return [("premium_discount", 0.9), ("gift_voucher", 0.85),
               ("cashback", 0.8)], "Win-Back Campaign"

    def recommend_reward(self, customer_id: str) -> Dict[str, Any]:
        """Recommend personalized reward/incentive using multi-armed bandit approach"""
        customer_id = validate_customer_id(customer_id)
        customer = self._get_customer(customer_id)

        if not customer:
            raise CustomerNotFoundError(f"Customer not found: {customer_id}")

        segmentation = self.segment_customer(customer_id)
        churn_prob = segmentation['churn_probability']
        detailed_segment = segmentation['detailed_segment']

        # Get reward strategy
        recommended_rewards, strategy = self._select_reward_strategy(detailed_segment, churn_prob)

        # Select top reward
        if recommended_rewards:
            top_reward_key, confidence = recommended_rewards[0]
            top_reward = REWARD_CATALOG[top_reward_key]
        else:
            top_reward_key = "standard_discount"
            top_reward = REWARD_CATALOG[top_reward_key]
            confidence = 0.5
            strategy = "Default Reward"

        # Calculate expected ROI
        expected_retention_lift = confidence * MAX_RETENTION_LIFT
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
                {"reward": REWARD_CATALOG[r[0]]['name'], "confidence": r[1]}
                for r in recommended_rewards[1:3]
            ] if len(recommended_rewards) > 1 else [],
            "expected_retention_lift": f"{expected_retention_lift*100:.1f}%",
            "expected_roi": f"{expected_roi:.1f}%",
            "reasoning": {
                "segment": detailed_segment,
                "churn_risk": "High" if churn_prob >= CHURN_RISK_HIGH else "Medium" if churn_prob >= CHURN_RISK_MEDIUM else "Low",
                "rfm_score": segmentation['rfm_score'],
                "lifetime_value": customer_ltv
            }
        }

    def analyze_customer(self, customer_id: str) -> Dict[str, Any]:
        """Comprehensive customer analysis combining all insights"""
        customer_id = validate_customer_id(customer_id)
        customer = self._get_customer(customer_id)

        if not customer:
            self.logger.warning(f"Customer not found: {customer_id}")
            raise CustomerNotFoundError(f"Customer not found: {customer_id}")

        rfm = self.calculate_rfm_score(customer_id)
        segmentation = self.segment_customer(customer_id)
        churn_prediction = self.predict_churn_probability(customer_id)
        reward_recommendation = self.recommend_reward(customer_id)

        self.logger.info(f"Analyzed customer {customer_id}: segment={segmentation['detailed_segment']}, "
                        f"churn={churn_prediction}")

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
                "risk_level": ("High" if churn_prediction >= CHURN_RISK_HIGH else
                             "Medium" if churn_prediction >= CHURN_RISK_MEDIUM else "Low"),
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

    def batch_analyze(self, customer_ids: Optional[List[str]] = None,
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Analyze multiple customers in batch"""
        if customer_ids:
            ids_to_analyze = [validate_customer_id(cid) for cid in customer_ids]
        else:
            ids_to_analyze = [c['customer_id'] for c in self.customers]

        if limit:
            limit = validate_limit(limit)
            ids_to_analyze = ids_to_analyze[:limit]

        self.logger.info(f"Batch analyzing {len(ids_to_analyze)} customers")

        results = []
        for customer_id in ids_to_analyze:
            analysis = self.analyze_customer(customer_id)
            results.append(analysis)

        return results

    def get_high_value_at_risk_customers(self, threshold: float = HIGH_VALUE_CHURN_THRESHOLD,
                                        min_ltv: float = HIGH_VALUE_MIN_LTV) -> List[Dict]:
        """Identify high-value customers at risk of churning"""
        threshold = validate_probability(threshold, "threshold")
        min_ltv = validate_probability(min_ltv, "min_ltv") if min_ltv <= 1 else min_ltv

        at_risk_customers = []

        for customer in self.customers:
            if customer['lifetime_value'] >= min_ltv:
                customer_id = customer['customer_id']
                churn_prob = self.predict_churn_probability(customer_id)

                if churn_prob >= threshold:
                    analysis = self.analyze_customer(customer_id)
                    at_risk_customers.append(analysis)

        at_risk_customers.sort(key=lambda x: x['kpis']['lifetime_value'], reverse=True)

        self.logger.info(f"Found {len(at_risk_customers)} high-value at-risk customers")

        return at_risk_customers


def main():
    """Demo: Analyze sample customers"""
    print("=" * 60)
    print("LOYALTY AI AGENT - DEMO")
    print("=" * 60)

    agent = LoyaltyAgent()

    if not agent.customers:
        print("\n⚠ No customer data found. Please run data_generator.py first.")
        return

    print(f"\nAnalyzing sample customers...\n")

    for customer in agent.customers[:5]:
        customer_id = customer['customer_id']
        analysis = agent.analyze_customer(customer_id)

        print(f"\n{'─' * 60}")
        print(f"Customer: {customer_id} | Segment: {analysis['segmentation']['detailed_segment']}")
        print(f"{'─' * 60}")
        print(f"RFM Score: {analysis['rfm_analysis']['rfm_score']}/100")
        print(f"Churn Risk: {analysis['churn_prediction']['risk_level']} "
              f"({analysis['churn_prediction']['probability']})")
        print(f"Lifetime Value: PKR {analysis['kpis']['lifetime_value']:,.2f}")
        print(f"\nRecommended Action: {analysis['recommendation']['strategy']}")
        print(f"Reward: {analysis['recommendation']['reward_details']['name']}")
        print(f"Expected ROI: {analysis['recommendation']['expected_roi']}")

    print(f"\n\n{'=' * 60}")
    print("HIGH-VALUE AT-RISK CUSTOMERS")
    print(f"{'=' * 60}")

    at_risk = agent.get_high_value_at_risk_customers(threshold=0.6, min_ltv=50000)
    print(f"\nFound {len(at_risk)} high-value customers at risk of churning\n")

    for i, customer_analysis in enumerate(at_risk[:3], 1):
        print(f"{i}. {customer_analysis['customer_id']}")
        print(f"   LTV: PKR {customer_analysis['kpis']['lifetime_value']:,.2f}")
        print(f"   Churn Risk: {customer_analysis['churn_prediction']['probability']}")
        print(f"   Action: {customer_analysis['recommendation']['reward_details']['name']}\n")

    print("=" * 60)


if __name__ == "__main__":
    main()
