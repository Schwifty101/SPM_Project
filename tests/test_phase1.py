"""
Unit tests for Phase 1: Data Generator and Loyalty Agent
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import CustomerDataGenerator
from src.loyalty_agent import LoyaltyAgent
from src.validators import validate_customer_id, validate_probability, ValidationError
from src.constants import REWARD_CATALOG, RFM_CHAMPION_THRESHOLD


class TestDataGenerator:
    """Test suite for CustomerDataGenerator"""

    def test_generator_initialization(self):
        """Test that generator initializes correctly"""
        generator = CustomerDataGenerator(num_customers=10, num_transactions=50, seed=42)
        assert generator.num_customers == 10
        assert generator.num_transactions == 50
        print("✓ Generator initialization test passed")

    def test_customer_generation(self):
        """Test customer profile generation"""
        generator = CustomerDataGenerator(num_customers=5, num_transactions=10, seed=42)
        customers, transactions = generator.generate_all_data()

        assert len(customers) == 5, f"Expected 5 customers, got {len(customers)}"
        assert len(transactions) == 10, f"Expected 10 transactions, got {len(transactions)}"

        # Check customer structure
        customer = customers[0]
        required_fields = ['customer_id', 'segment', 'loyalty_tier', 'registration_date',
                          'lifetime_value', 'engagement_score']
        for field in required_fields:
            assert field in customer, f"Missing field: {field}"

        print("✓ Customer generation test passed")

    def test_transaction_generation(self):
        """Test transaction generation"""
        generator = CustomerDataGenerator(num_customers=5, num_transactions=10, seed=42)
        customers, transactions = generator.generate_all_data()

        # Check transaction structure
        transaction = transactions[0]
        required_fields = ['transaction_id', 'customer_id', 'timestamp',
                          'product_category', 'final_amount', 'status']
        for field in required_fields:
            assert field in transaction, f"Missing field: {field}"

        # Verify customer_id references exist
        customer_ids = {c['customer_id'] for c in customers}
        for txn in transactions:
            assert txn['customer_id'] in customer_ids, \
                f"Transaction references non-existent customer: {txn['customer_id']}"

        print("✓ Transaction generation test passed")


class TestValidators:
    """Test suite for validation utilities"""

    def test_validate_customer_id(self):
        """Test customer ID validation"""
        # Valid cases
        assert validate_customer_id("CUST123") == "CUST123"
        assert validate_customer_id("  CUST456  ") == "CUST456"

        # Invalid cases
        try:
            validate_customer_id(None)
            assert False, "Should raise ValidationError for None"
        except ValidationError:
            pass

        try:
            validate_customer_id("")
            assert False, "Should raise ValidationError for empty string"
        except ValidationError:
            pass

        print("✓ Customer ID validation test passed")

    def test_validate_probability(self):
        """Test probability validation"""
        # Valid cases
        assert validate_probability(0.5) == 0.5
        assert validate_probability(0) == 0.0
        assert validate_probability(1.0) == 1.0

        # Invalid cases
        try:
            validate_probability(-0.1)
            assert False, "Should raise ValidationError for negative value"
        except ValidationError:
            pass

        try:
            validate_probability(1.5)
            assert False, "Should raise ValidationError for value > 1"
        except ValidationError:
            pass

        print("✓ Probability validation test passed")


class TestLoyaltyAgent:
    """Test suite for LoyaltyAgent"""

    @classmethod
    def setup_test_data(cls):
        """Generate test data for agent tests"""
        generator = CustomerDataGenerator(num_customers=100, num_transactions=500, seed=42)
        customers, transactions = generator.generate_all_data()
        generator.save_to_json(customers, transactions, output_dir="data")
        return customers[0]['customer_id']

    def test_agent_initialization(self):
        """Test that agent initializes and loads data"""
        agent = LoyaltyAgent()
        assert len(agent.customers) > 0, "No customers loaded"
        assert len(agent.transactions) > 0, "No transactions loaded"
        print(f"✓ Agent initialization test passed (loaded {len(agent.customers)} customers)")

    def test_rfm_calculation(self):
        """Test RFM score calculation"""
        customer_id = self.setup_test_data()
        agent = LoyaltyAgent()

        rfm = agent.calculate_rfm_score(customer_id)

        assert 'rfm_score' in rfm, "Missing rfm_score"
        assert 0 <= rfm['rfm_score'] <= 100, f"RFM score out of range: {rfm['rfm_score']}"
        assert 'recency' in rfm, "Missing recency score"
        assert 'frequency' in rfm, "Missing frequency score"
        assert 'monetary' in rfm, "Missing monetary score"

        print(f"✓ RFM calculation test passed (score: {rfm['rfm_score']})")

    def test_churn_prediction(self):
        """Test churn probability prediction"""
        customer_id = self.setup_test_data()
        agent = LoyaltyAgent()

        churn_prob = agent.predict_churn_probability(customer_id)

        assert 0 <= churn_prob <= 1, f"Churn probability out of range: {churn_prob}"
        print(f"✓ Churn prediction test passed (probability: {churn_prob})")

    def test_customer_segmentation(self):
        """Test customer segmentation"""
        customer_id = self.setup_test_data()
        agent = LoyaltyAgent()

        segmentation = agent.segment_customer(customer_id)

        assert 'detailed_segment' in segmentation, "Missing detailed_segment"
        assert 'rfm_score' in segmentation, "Missing rfm_score"
        assert 'churn_probability' in segmentation, "Missing churn_probability"
        assert 'is_at_risk' in segmentation, "Missing is_at_risk flag"

        print(f"✓ Customer segmentation test passed (segment: {segmentation['detailed_segment']})")

    def test_reward_recommendation(self):
        """Test reward recommendation"""
        customer_id = self.setup_test_data()
        agent = LoyaltyAgent()

        recommendation = agent.recommend_reward(customer_id)

        assert 'recommended_reward' in recommendation, "Missing recommended_reward"
        assert recommendation['recommended_reward'] in REWARD_CATALOG, \
            f"Invalid reward: {recommendation['recommended_reward']}"
        assert 'confidence' in recommendation, "Missing confidence"
        assert 'strategy' in recommendation, "Missing strategy"
        assert 'expected_roi' in recommendation, "Missing expected_roi"

        print(f"✓ Reward recommendation test passed (reward: {recommendation['recommended_reward']})")

    def test_full_customer_analysis(self):
        """Test complete customer analysis"""
        customer_id = self.setup_test_data()
        agent = LoyaltyAgent()

        analysis = agent.analyze_customer(customer_id)

        assert 'customer_id' in analysis, "Missing customer_id"
        assert 'profile' in analysis, "Missing profile"
        assert 'rfm_analysis' in analysis, "Missing rfm_analysis"
        assert 'segmentation' in analysis, "Missing segmentation"
        assert 'churn_prediction' in analysis, "Missing churn_prediction"
        assert 'recommendation' in analysis, "Missing recommendation"
        assert 'kpis' in analysis, "Missing kpis"

        print("✓ Full customer analysis test passed")

    def test_batch_analysis(self):
        """Test batch customer analysis"""
        self.setup_test_data()
        agent = LoyaltyAgent()

        results = agent.batch_analyze(limit=5)

        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        for result in results:
            assert 'customer_id' in result, "Missing customer_id in batch result"

        print(f"✓ Batch analysis test passed (analyzed {len(results)} customers)")

    def test_high_value_at_risk(self):
        """Test high-value at-risk customer identification"""
        self.setup_test_data()
        agent = LoyaltyAgent()

        at_risk = agent.get_high_value_at_risk_customers(threshold=0.5, min_ltv=10000)

        # Should return a list (may be empty)
        assert isinstance(at_risk, list), "Expected list result"

        if at_risk:
            # Check structure if results exist
            customer = at_risk[0]
            assert customer['kpis']['lifetime_value'] >= 10000, \
                "Customer LTV below threshold"
            assert customer['churn_prediction']['probability'] >= 0.5, \
                "Customer churn probability below threshold"

        print(f"✓ High-value at-risk test passed (found {len(at_risk)} customers)")

    def test_invalid_customer_handling(self):
        """Test handling of invalid customer IDs"""
        from src.validators import CustomerNotFoundError
        agent = LoyaltyAgent()

        # Test with non-existent customer - should raise exception
        try:
            agent.analyze_customer("INVALID999")
            assert False, "Should raise CustomerNotFoundError for invalid customer"
        except CustomerNotFoundError as e:
            assert "INVALID999" in str(e), "Exception should contain customer ID"

        print("✓ Invalid customer handling test passed")


def run_all_tests():
    """Run all test suites"""
    print("=" * 70)
    print("PHASE 1 TEST SUITE - Data Generator & Loyalty Agent")
    print("=" * 70)

    test_suites = [
        ("Data Generator Tests", TestDataGenerator()),
        ("Validator Tests", TestValidators()),
        ("Loyalty Agent Tests", TestLoyaltyAgent())
    ]

    total_tests = 0
    passed_tests = 0

    for suite_name, test_suite in test_suites:
        print(f"\n{suite_name}")
        print("-" * 70)

        test_methods = [method for method in dir(test_suite) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_suite, test_method)()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {test_method} FAILED: {e}")
            except Exception as e:
                print(f"✗ {test_method} ERROR: {e}")

    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)

    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
