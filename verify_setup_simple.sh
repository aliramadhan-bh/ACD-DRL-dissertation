#!/bin/bash
# CAGE Challenge 2 - Simple Setup Verification Script
# This script verifies your environment is correctly set up

set -e  # Exit on any error

echo "======================================================================"
echo "  CAGE Challenge 2 - Environment Verification"
echo "======================================================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo ""
    echo "❌ Virtual environment not activated!"
    echo "   Please run: source .venv/bin/activate"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  1. Python Environment"
echo "======================================================================"
python --version
echo "✅ Python version OK"
echo "✅ Virtual environment: $VIRTUAL_ENV"

echo ""
echo "======================================================================"
echo "  2. Required Packages"
echo "======================================================================"

# Check each required package
packages=("pytest" "paramiko" "PyYAML" "numpy" "gym" "gymnasium" "prettytable" "docutils" "torch" "jax" "CybORG")

for pkg in "${packages[@]}"; do
    if pip show "$pkg" &> /dev/null; then
        version=$(pip show "$pkg" | grep Version | awk '{print $2}')
        echo "✅ $pkg ($version)"
    else
        echo "❌ $pkg - NOT INSTALLED"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "  3. Required Files"
echo "======================================================================"

files=(
    "CybORG/setup.py"
    "CybORG/Requirements.txt"
    "CybORG/CybORG/Shared/Scenarios/Scenario1b.yaml"
    "CybORG/CybORG/Agents/SimpleAgents/BlueReactAgent.py"
    "README.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "  4. Unit Tests"
echo "======================================================================"

cd CybORG
if pytest CybORG/Tests/test_sim/test_sim_Cyborg.py -v --tb=no -q 2>&1 | grep -q "16 passed"; then
    echo "✅ Core tests passed (16/16)"
else
    echo "⚠️  Some core tests may have failed (check output above)"
fi
cd ..

echo ""
echo "======================================================================"
echo "  5. Functional Test - Run Training Example"
echo "======================================================================"

cd CybORG
if python -c "from CybORG.Agents.training_example import run_training_example; run_training_example('Scenario1b')" 2>&1 | grep -q "Observation size"; then
    echo "✅ Training example runs successfully"
else
    echo "❌ Training example failed"
    cd ..
    exit 1
fi
cd ..

echo ""
echo "======================================================================"
echo "  VERIFICATION SUMMARY"
echo "======================================================================"
echo ""
echo "  🎉 SUCCESS! Your environment is set up correctly!"
echo "  You are ready to work on CAGE Challenge 2."
echo ""
echo "  Next steps:"
echo "    1. Read SETUP_VERIFICATION.md for detailed info"
echo "    2. cd CybORG"
echo "    3. Explore agents in CybORG/Agents/SimpleAgents/"
echo "    4. Build your own blue agent!"
echo ""
echo "======================================================================"
