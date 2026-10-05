# LeakGuard AI — Enhanced MVP

Run:
```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Features:
- Transaction anomaly detection
- Duplicate transaction detection
- Supplier price anomaly detection
- Quantity anomaly detection
- Isolation Forest ML detection
- Risk scoring with High/Critical cases
- Vendor intelligence
- Inventory reconciliation
- AI-style investigation assistant
- Interactive Plotly charts
- Downloadable leakage review CSV
- Upload your own transaction/inventory CSVs

Transaction CSV required columns:
`transaction_id, vendor_id, vendor_name, product_id, quantity, unit_price, total_amount`

Inventory CSV required columns:
`product_id, opening_stock, purchased_quantity, received_quantity, sold_quantity, closing_stock`

This prototype identifies potential leakage and supports human review; it does not prove fraud.
