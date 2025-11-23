"""
Data Generator for Loyalty AI Agent
Generates synthetic customer profiles and transaction data for testing and demonstration
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np


class CustomerDataGenerator:
    """Generates synthetic customer profiles and transaction data"""

    # Customer segments
    SEGMENTS = ['Premium', 'Regular', 'Occasional', 'New']

    # Loyalty tiers
    LOYALTY_TIERS = ['Gold', 'Silver', 'Bronze', 'Standard']

    # Product categories
    PRODUCT_CATEGORIES = [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports',
        'Beauty', 'Books', 'Food & Beverage', 'Toys', 'Automotive'
    ]

    # Product items (sample products per category)
    PRODUCTS = {
        'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smart Watch'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Dress'],
        'Home & Garden': ['Furniture', 'Decor', 'Kitchen Appliance', 'Bedding', 'Tools'],
        'Sports': ['Running Shoes', 'Yoga Mat', 'Dumbbell Set', 'Bicycle', 'Sports Wear'],
        'Beauty': ['Skincare', 'Makeup', 'Hair Care', 'Fragrance', 'Nail Care'],
        'Books': ['Fiction Novel', 'Non-Fiction', 'Educational', 'Comics', 'Magazine'],
        'Food & Beverage': ['Snacks', 'Beverages', 'Organic Foods', 'Frozen Items', 'Bakery'],
        'Toys': ['Action Figures', 'Board Games', 'Puzzles', 'Educational Toys', 'Dolls'],
        'Automotive': ['Car Accessories', 'Motor Oil', 'Tires', 'Cleaning Products', 'Tools']
    }

    def __init__(self, num_customers: int = 1000, num_transactions: int = 10000, seed: int = 42):
        """
        Initialize data generator

        Args:
            num_customers: Number of customer profiles to generate
            num_transactions: Number of transaction records to generate
            seed: Random seed for reproducibility
        """
        self.num_customers = num_customers
        self.num_transactions = num_transactions
        random.seed(seed)
        np.random.seed(seed)

        # Date range for transactions (last 2 years)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=730)

    def generate_customer_id(self, index: int) -> str:
        """Generate unique customer ID"""
        return f"CUST{index:06d}"

    def generate_transaction_id(self, index: int) -> str:
        """Generate unique transaction ID"""
        return f"TXN{index:08d}"

    def generate_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Generate a single customer profile with realistic attributes

        Args:
            customer_id: Unique customer identifier

        Returns:
            Dictionary containing customer profile data
        """
        # Segment distribution: Premium (10%), Regular (30%), Occasional (40%), New (20%)
        segment_weights = [0.10, 0.30, 0.40, 0.20]
        segment = random.choices(self.SEGMENTS, weights=segment_weights)[0]

        # Loyalty tier based on segment
        if segment == 'Premium':
            loyalty_tier = random.choice(['Gold', 'Silver'])
        elif segment == 'Regular':
            loyalty_tier = random.choice(['Silver', 'Bronze'])
        elif segment == 'Occasional':
            loyalty_tier = random.choice(['Bronze', 'Standard'])
        else:  # New
            loyalty_tier = 'Standard'

        # Registration date (customers joined over last 2 years)
        if segment == 'New':
            # New customers joined in last 3 months
            days_ago = random.randint(1, 90)
        elif segment == 'Occasional':
            days_ago = random.randint(91, 365)
        else:
            days_ago = random.randint(366, 730)

        registration_date = self.end_date - timedelta(days=days_ago)

        # Purchase history summary
        if segment == 'Premium':
            total_purchases = random.randint(50, 150)
            avg_order_value = random.uniform(500, 2000)
            purchase_frequency = random.uniform(2, 4)  # purchases per month
        elif segment == 'Regular':
            total_purchases = random.randint(20, 60)
            avg_order_value = random.uniform(200, 800)
            purchase_frequency = random.uniform(1, 2.5)
        elif segment == 'Occasional':
            total_purchases = random.randint(5, 25)
            avg_order_value = random.uniform(100, 500)
            purchase_frequency = random.uniform(0.3, 1.2)
        else:  # New
            total_purchases = random.randint(1, 5)
            avg_order_value = random.uniform(50, 300)
            purchase_frequency = random.uniform(0.1, 0.5)

        # Calculate lifetime value
        lifetime_value = total_purchases * avg_order_value

        # Last purchase date
        last_purchase_days = random.randint(1, 180) if segment in ['Premium', 'Regular'] else random.randint(30, 365)
        last_purchase_date = self.end_date - timedelta(days=last_purchase_days)

        # Preferred categories (weighted by segment)
        num_preferred_categories = random.randint(2, 4)
        preferred_categories = random.sample(self.PRODUCT_CATEGORIES, num_preferred_categories)

        # Engagement score (0-100)
        if segment == 'Premium':
            engagement_score = random.uniform(70, 100)
        elif segment == 'Regular':
            engagement_score = random.uniform(50, 80)
        elif segment == 'Occasional':
            engagement_score = random.uniform(20, 60)
        else:
            engagement_score = random.uniform(0, 40)

        # Churn risk (inverse of engagement)
        churn_risk = round(1 - (engagement_score / 100), 2)

        return {
            "customer_id": customer_id,
            "segment": segment,
            "loyalty_tier": loyalty_tier,
            "registration_date": registration_date.strftime("%Y-%m-%d"),
            "last_purchase_date": last_purchase_date.strftime("%Y-%m-%d"),
            "total_purchases": total_purchases,
            "avg_order_value": round(avg_order_value, 2),
            "lifetime_value": round(lifetime_value, 2),
            "purchase_frequency": round(purchase_frequency, 2),
            "preferred_categories": preferred_categories,
            "engagement_score": round(engagement_score, 2),
            "churn_risk": churn_risk,
            "contact_email": f"{customer_id.lower()}@example.com",
            "is_active": last_purchase_days < 180
        }

    def generate_transaction(self, transaction_id: str, customer_profiles: List[Dict]) -> Dict[str, Any]:
        """
        Generate a single transaction record

        Args:
            transaction_id: Unique transaction identifier
            customer_profiles: List of customer profiles for reference

        Returns:
            Dictionary containing transaction data
        """
        # Select random customer (weighted by segment)
        customer = random.choice(customer_profiles)
        customer_segment = customer['segment']

        # Transaction date (within last 2 years, weighted recent)
        # More recent transactions are more likely
        days_ago = int(np.random.exponential(scale=180))  # Exponential distribution
        days_ago = min(days_ago, 730)  # Cap at 2 years
        transaction_date = self.end_date - timedelta(days=days_ago)

        # Ensure transaction is after customer registration
        registration_date = datetime.strptime(customer['registration_date'], "%Y-%m-%d")
        if transaction_date < registration_date:
            transaction_date = registration_date + timedelta(days=random.randint(1, 30))

        # Select product category (prefer customer's preferred categories)
        if random.random() < 0.7:  # 70% chance of preferred category
            category = random.choice(customer['preferred_categories'])
        else:
            category = random.choice(self.PRODUCT_CATEGORIES)

        # Select product from category
        product = random.choice(self.PRODUCTS[category])

        # Transaction amount based on customer segment
        if customer_segment == 'Premium':
            amount = random.uniform(300, 2500)
        elif customer_segment == 'Regular':
            amount = random.uniform(150, 1000)
        elif customer_segment == 'Occasional':
            amount = random.uniform(50, 600)
        else:  # New
            amount = random.uniform(20, 400)

        # Quantity
        quantity = random.randint(1, 5)

        # Discount applied (Premium customers get more discounts)
        if customer_segment == 'Premium':
            discount_applied = random.choice([True, True, True, False])  # 75% chance
            if discount_applied:
                discount_amount = round(amount * random.uniform(0.1, 0.3), 2)
            else:
                discount_amount = 0
        elif customer_segment == 'Regular':
            discount_applied = random.choice([True, False])  # 50% chance
            if discount_applied:
                discount_amount = round(amount * random.uniform(0.05, 0.2), 2)
            else:
                discount_amount = 0
        else:
            discount_applied = random.choice([True, False, False, False])  # 25% chance
            if discount_applied:
                discount_amount = round(amount * random.uniform(0.05, 0.15), 2)
            else:
                discount_amount = 0

        final_amount = round(amount - discount_amount, 2)

        # Payment method
        payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Gift Card', 'Cash']
        payment_method = random.choice(payment_methods)

        # Transaction status (99% successful)
        status = random.choices(['Completed', 'Failed'], weights=[0.99, 0.01])[0]

        return {
            "transaction_id": transaction_id,
            "customer_id": customer['customer_id'],
            "timestamp": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
            "product_category": category,
            "product_name": product,
            "quantity": quantity,
            "original_amount": round(amount, 2),
            "discount_applied": discount_applied,
            "discount_amount": discount_amount,
            "final_amount": final_amount,
            "payment_method": payment_method,
            "status": status
        }

    def generate_all_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate all customer profiles and transactions

        Returns:
            Tuple of (customer_profiles, transactions)
        """
        print(f"Generating {self.num_customers} customer profiles...")
        customers = []
        for i in range(self.num_customers):
            customer_id = self.generate_customer_id(i)
            customer = self.generate_customer_profile(customer_id)
            customers.append(customer)

        print(f"Generating {self.num_transactions} transaction records...")
        transactions = []
        for i in range(self.num_transactions):
            transaction_id = self.generate_transaction_id(i)
            transaction = self.generate_transaction(transaction_id, customers)
            transactions.append(transaction)

        return customers, transactions

    def save_to_json(self, customers: List[Dict], transactions: List[Dict], output_dir: str = "data"):
        """
        Save generated data to JSON files

        Args:
            customers: List of customer profiles
            transactions: List of transactions
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save customers
        customers_file = output_path / "customers.json"
        with open(customers_file, 'w') as f:
            json.dump(customers, f, indent=2)
        print(f"✓ Saved {len(customers)} customers to {customers_file}")

        # Save transactions
        transactions_file = output_path / "transactions.json"
        with open(transactions_file, 'w') as f:
            json.dump(transactions, f, indent=2)
        print(f"✓ Saved {len(transactions)} transactions to {transactions_file}")

        # Generate summary statistics
        self.print_summary(customers, transactions)

    def print_summary(self, customers: List[Dict], transactions: List[Dict]):
        """Print summary statistics of generated data"""
        print("\n" + "="*50)
        print("DATA GENERATION SUMMARY")
        print("="*50)

        # Customer statistics
        segments = {}
        tiers = {}
        for customer in customers:
            segment = customer['segment']
            tier = customer['loyalty_tier']
            segments[segment] = segments.get(segment, 0) + 1
            tiers[tier] = tiers.get(tier, 0) + 1

        print("\nCustomer Distribution:")
        for segment, count in sorted(segments.items()):
            print(f"  {segment}: {count} ({count/len(customers)*100:.1f}%)")

        print("\nLoyalty Tier Distribution:")
        for tier, count in sorted(tiers.items()):
            print(f"  {tier}: {count} ({count/len(customers)*100:.1f}%)")

        # Transaction statistics
        completed_txns = [t for t in transactions if t['status'] == 'Completed']
        total_revenue = sum(t['final_amount'] for t in completed_txns)
        avg_transaction = total_revenue / len(completed_txns) if completed_txns else 0

        print(f"\nTransaction Statistics:")
        print(f"  Total Transactions: {len(transactions)}")
        print(f"  Completed: {len(completed_txns)} ({len(completed_txns)/len(transactions)*100:.1f}%)")
        print(f"  Total Revenue: PKR {total_revenue:,.2f}")
        print(f"  Average Transaction: PKR {avg_transaction:,.2f}")

        # Category distribution
        categories = {}
        for txn in transactions:
            cat = txn['product_category']
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\nTop Product Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {cat}: {count} transactions")

        print("="*50 + "\n")


def main():
    """Main execution function"""
    # Initialize generator
    generator = CustomerDataGenerator(
        num_customers=1000,
        num_transactions=10000,
        seed=42
    )

    # Generate data
    customers, transactions = generator.generate_all_data()

    # Save to JSON files
    generator.save_to_json(customers, transactions, output_dir="data")

    print("✓ Data generation completed successfully!")


if __name__ == "__main__":
    main()
