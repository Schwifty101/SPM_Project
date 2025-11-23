"""
Comprehensive Phase 1 Testing Script
Tests all core functionality of the Loyalty AI Agent
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loyalty_agent import LoyaltyAgent

def test_data_exists():
    """Test 1: Verify data files exist and are valid"""
    print("="*70)
    print("TEST 1: Data Files Validation")
    print("="*70)

    customers_file = Path("data/customers.json")
    transactions_file = Path("data/transactions.json")

    # Check files exist
    assert customers_file.exists(), "❌ customers.json not found"
    print("✅ customers.json exists")

    assert transactions_file.exists(), "❌ transactions.json not found"
    print("✅ transactions.json exists")

    # Check files are valid JSON
    with open(customers_file) as f:
        customers = json.load(f)
    print(f"✅ Loaded {len(customers)} customers")

    with open(transactions_file) as f:
        transactions = json.load(f)
    print(f"✅ Loaded {len(transactions)} transactions")

    # Verify data counts match plan
    assert len(customers) == 1000, f"❌ Expected 1000 customers, got {len(customers)}"
    print("✅ Customer count matches plan (1000)")

    assert len(transactions) == 10000, f"❌ Expected 10000 transactions, got {len(transactions)}"
    print("✅ Transaction count matches plan (10000)")

    return customers, transactions


def test_customer_segments(customers):
    """Test 2: Verify customer segment distribution"""
    print("\n" + "="*70)
    print("TEST 2: Customer Segment Distribution")
    print("="*70)

    segments = {}
    tiers = {}

    for customer in customers:
        seg = customer['segment']
        tier = customer['loyalty_tier']
        segments[seg] = segments.get(seg, 0) + 1
        tiers[tier] = tiers.get(tier, 0) + 1

    print("\nSegment Distribution:")
    for segment, count in sorted(segments.items()):
        pct = count / len(customers) * 100
        print(f"  {segment}: {count} ({pct:.1f}%)")

    print("\nLoyalty Tier Distribution:")
    for tier, count in sorted(tiers.items()):
        pct = count / len(customers) * 100
        print(f"  {tier}: {count} ({pct:.1f}%)")

    # Verify all segments exist
    required_segments = ['Premium', 'Regular', 'Occasional', 'New']
    for seg in required_segments:
        assert seg in segments, f"❌ Missing segment: {seg}"
    print("\n✅ All required segments present")

    # Verify all tiers exist
    required_tiers = ['Gold', 'Silver', 'Bronze', 'Standard']
    for tier in required_tiers:
        assert tier in tiers, f"❌ Missing tier: {tier}"
    print("✅ All loyalty tiers present")


def test_rfm_analysis(agent):
    """Test 3: RFM Analysis functionality"""
    print("\n" + "="*70)
    print("TEST 3: RFM Analysis")
    print("="*70)

    # Test with different customer types
    test_customers = agent.customers[:10]

    for customer in test_customers:
        customer_id = customer['customer_id']
        rfm = agent.calculate_rfm_score(customer_id)

        # Verify RFM structure
        assert 'recency' in rfm, "❌ Missing 'recency' in RFM"
        assert 'frequency' in rfm, "❌ Missing 'frequency' in RFM"
        assert 'monetary' in rfm, "❌ Missing 'monetary' in RFM"
        assert 'rfm_score' in rfm, "❌ Missing 'rfm_score' in RFM"

        # Verify scores are in valid range (0-100)
        assert 0 <= rfm['recency'] <= 100, f"❌ Invalid recency score: {rfm['recency']}"
        assert 0 <= rfm['frequency'] <= 100, f"❌ Invalid frequency score: {rfm['frequency']}"
        assert 0 <= rfm['monetary'] <= 100, f"❌ Invalid monetary score: {rfm['monetary']}"
        assert 0 <= rfm['rfm_score'] <= 100, f"❌ Invalid RFM score: {rfm['rfm_score']}"

    print(f"✅ RFM analysis tested on {len(test_customers)} customers")
    print(f"✅ All RFM scores are in valid range (0-100)")
    print(f"✅ Sample RFM Score: {test_customers[0]['customer_id']} = {agent.calculate_rfm_score(test_customers[0]['customer_id'])['rfm_score']}")


def test_churn_prediction(agent):
    """Test 4: Churn Prediction functionality"""
    print("\n" + "="*70)
    print("TEST 4: Churn Prediction")
    print("="*70)

    test_customers = agent.customers[:20]

    high_risk = 0
    medium_risk = 0
    low_risk = 0

    for customer in test_customers:
        customer_id = customer['customer_id']
        churn_prob = agent.predict_churn_probability(customer_id)

        # Verify churn probability is valid (0-1)
        assert 0 <= churn_prob <= 1, f"❌ Invalid churn probability: {churn_prob}"

        if churn_prob >= 0.7:
            high_risk += 1
        elif churn_prob >= 0.4:
            medium_risk += 1
        else:
            low_risk += 1

    print(f"✅ Churn prediction tested on {len(test_customers)} customers")
    print(f"  High Risk (>=0.7): {high_risk}")
    print(f"  Medium Risk (0.4-0.7): {medium_risk}")
    print(f"  Low Risk (<0.4): {low_risk}")
    print("✅ All churn probabilities are in valid range (0-1)")


def test_customer_segmentation(agent):
    """Test 5: Customer Segmentation"""
    print("\n" + "="*70)
    print("TEST 5: Customer Segmentation")
    print("="*70)

    test_customers = agent.customers[:15]

    segments_found = set()

    for customer in test_customers:
        customer_id = customer['customer_id']
        segmentation = agent.segment_customer(customer_id)

        # Verify segmentation structure
        assert 'detailed_segment' in segmentation, "❌ Missing 'detailed_segment'"
        assert 'rfm_score' in segmentation, "❌ Missing 'rfm_score'"
        assert 'churn_probability' in segmentation, "❌ Missing 'churn_probability'"
        assert 'is_at_risk' in segmentation, "❌ Missing 'is_at_risk'"

        segments_found.add(segmentation['detailed_segment'])

    print(f"✅ Segmentation tested on {len(test_customers)} customers")
    print(f"✅ Segments found: {', '.join(sorted(segments_found))}")

    # Valid segments
    valid_segments = [
        "Champion", "Loyal Customer", "At-Risk Champion", "At-Risk Loyal",
        "Potential Loyalist", "Hibernating", "New Customer", "Lost Customer"
    ]

    for seg in segments_found:
        assert seg in valid_segments, f"❌ Invalid segment: {seg}"

    print("✅ All segments are valid")


def test_reward_recommendations(agent):
    """Test 6: Reward Recommendations"""
    print("\n" + "="*70)
    print("TEST 6: Reward Recommendations")
    print("="*70)

    test_customers = agent.customers[:10]

    reward_types_found = set()
    strategies_found = set()

    for customer in test_customers:
        customer_id = customer['customer_id']
        recommendation = agent.recommend_reward(customer_id)

        # Verify recommendation structure
        assert 'recommended_reward' in recommendation, "❌ Missing 'recommended_reward'"
        assert 'reward_details' in recommendation, "❌ Missing 'reward_details'"
        assert 'confidence' in recommendation, "❌ Missing 'confidence'"
        assert 'strategy' in recommendation, "❌ Missing 'strategy'"
        assert 'expected_roi' in recommendation, "❌ Missing 'expected_roi'"

        # Verify confidence is valid (0-1)
        confidence = recommendation['confidence']
        assert 0 <= confidence <= 1, f"❌ Invalid confidence: {confidence}"

        reward_types_found.add(recommendation['recommended_reward'])
        strategies_found.add(recommendation['strategy'])

    print(f"✅ Reward recommendations tested on {len(test_customers)} customers")
    print(f"✅ Reward types used: {len(reward_types_found)}")
    print(f"  {', '.join(sorted(reward_types_found))}")
    print(f"✅ Strategies found: {len(strategies_found)}")
    print(f"  {', '.join(sorted(strategies_found))}")


def test_comprehensive_analysis(agent):
    """Test 7: Comprehensive Customer Analysis"""
    print("\n" + "="*70)
    print("TEST 7: Comprehensive Analysis")
    print("="*70)

    customer_id = agent.customers[0]['customer_id']
    analysis = agent.analyze_customer(customer_id)

    # Verify comprehensive analysis structure
    required_keys = [
        'customer_id', 'profile', 'rfm_analysis', 'segmentation',
        'churn_prediction', 'recommendation', 'kpis', 'timestamp'
    ]

    for key in required_keys:
        assert key in analysis, f"❌ Missing key in analysis: {key}"

    print(f"✅ Comprehensive analysis completed for {customer_id}")
    print(f"✅ All required components present")

    # Display sample analysis
    print(f"\nSample Analysis Summary:")
    print(f"  Segment: {analysis['segmentation']['detailed_segment']}")
    print(f"  RFM Score: {analysis['rfm_analysis']['rfm_score']}/100")
    print(f"  Churn Risk: {analysis['churn_prediction']['risk_level']}")
    print(f"  Recommended Reward: {analysis['recommendation']['reward_details']['name']}")
    print(f"  Expected ROI: {analysis['recommendation']['expected_roi']}")


def test_high_value_at_risk(agent):
    """Test 8: High-Value At-Risk Customers"""
    print("\n" + "="*70)
    print("TEST 8: High-Value At-Risk Customers")
    print("="*70)

    # Test with different thresholds
    at_risk_high = agent.get_high_value_at_risk_customers(threshold=0.6, min_ltv=50000)
    at_risk_medium = agent.get_high_value_at_risk_customers(threshold=0.5, min_ltv=30000)

    print(f"✅ High-value at-risk (LTV>50k, churn>0.6): {len(at_risk_high)} customers")
    print(f"✅ Medium-value at-risk (LTV>30k, churn>0.5): {len(at_risk_medium)} customers")

    if at_risk_high:
        sample = at_risk_high[0]
        print(f"\n  Sample High-Risk Customer:")
        print(f"    ID: {sample['customer_id']}")
        print(f"    LTV: PKR {sample['kpis']['lifetime_value']:,.2f}")
        print(f"    Churn Risk: {sample['churn_prediction']['probability']}")


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  PHASE 1 COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        # Test 1: Data exists
        customers, transactions = test_data_exists()

        # Test 2: Customer segments
        test_customer_segments(customers)

        # Initialize agent
        agent = LoyaltyAgent()

        # Test 3: RFM Analysis
        test_rfm_analysis(agent)

        # Test 4: Churn Prediction
        test_churn_prediction(agent)

        # Test 5: Customer Segmentation
        test_customer_segmentation(agent)

        # Test 6: Reward Recommendations
        test_reward_recommendations(agent)

        # Test 7: Comprehensive Analysis
        test_comprehensive_analysis(agent)

        # Test 8: High-Value At-Risk
        test_high_value_at_risk(agent)

        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - PHASE 1 COMPLETE")
        print("="*70)
        print("\n📊 Test Summary:")
        print("  ✅ Data Generation")
        print("  ✅ RFM Analysis")
        print("  ✅ Churn Prediction")
        print("  ✅ Customer Segmentation")
        print("  ✅ Reward Recommendations")
        print("  ✅ Comprehensive Analysis")
        print("  ✅ High-Value Customer Identification")
        print("\n🎯 Status: PHASE 1 READY FOR DEPLOYMENT")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
