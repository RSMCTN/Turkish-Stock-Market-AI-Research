# 🤝 Contributing to Turkish Stock Market AI Research

Thank you for your interest in contributing to this research project! This document provides guidelines for contributions.

## 🎯 Research Areas

We welcome contributions in the following areas:

### 🔬 **Machine Learning Research**
- DP-LSTM architecture improvements
- Turkish language NLP enhancements
- Multi-language sentiment analysis models
- Performance optimization techniques

### 📊 **Data Integration**
- New financial data sources
- Real-time data processing improvements
- API integration optimizations
- Data quality enhancements

### 💻 **Technical Development**
- Frontend UI/UX improvements
- Backend performance optimizations
- Database schema enhancements
- Deployment and DevOps improvements

### 📚 **Documentation**
- Research methodology documentation
- Technical architecture explanations
- Tutorial and example creation
- Translation to other languages

## 🚀 Getting Started

### Prerequisites
```bash
# Required software
- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis 6+
- Docker & Docker Compose
```

### Development Setup
```bash
# Clone the repository
git clone https://github.com/[username]/Turkish-Stock-Market-AI-Research.git
cd Turkish-Stock-Market-AI-Research

# Backend setup
pip install -r requirements.txt
cd src && uvicorn api.main:app --reload

# Frontend setup  
cd global-dashboard
npm install
npm run dev
```

## 📋 Contribution Process

### 1. **Issue Discussion**
- Check existing issues before creating new ones
- Use issue templates for bug reports and feature requests
- Participate in discussions for research directions
- Tag issues with appropriate labels

### 2. **Development Workflow**
```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Write tests if applicable
# Update documentation

# Commit with conventional commits
git commit -m "feat: add Turkish sentiment model improvement"

# Push and create pull request
git push origin feature/your-feature-name
```

### 3. **Pull Request Guidelines**
- Fill out the PR template completely
- Include relevant issue references
- Add screenshots for UI changes
- Ensure all tests pass
- Update documentation as needed

## 🔬 Research Contributions

### **Academic Research**
- Follow academic citation standards
- Include methodology documentation
- Provide reproducible experiments
- Share datasets when possible (following legal requirements)

### **Model Improvements**
- Document performance benchmarks
- Include training procedures
- Provide model evaluation metrics
- Consider privacy implications (DP-LSTM)

### **Data Research**
- Ensure data compliance (GDPR, KVKK, etc.)
- Document data sources and licensing
- Provide data quality assessments
- Include data preprocessing steps

## 🧪 Testing Guidelines

### **Unit Tests**
```bash
# Python backend tests
pytest tests/

# Frontend tests
cd global-dashboard && npm test
```

### **Integration Tests**
```bash
# Full system tests
docker-compose -f docker-compose.test.yml up --build
```

### **Performance Tests**
- API response time benchmarks
- Database query performance
- ML model inference speed
- Frontend loading performance

## 📊 Code Standards

### **Python (Backend)**
- Follow PEP 8 style guide
- Use type hints
- Document functions with docstrings
- Maximum line length: 88 characters
- Use Black for formatting
- Use pylint for linting

### **TypeScript/JavaScript (Frontend)**
- Use TypeScript for new code
- Follow ESLint configuration
- Use Prettier for formatting
- Document components with JSDoc
- Follow React best practices

### **Documentation**
- Use Markdown for documentation
- Include code examples
- Provide architectural diagrams
- Maintain up-to-date API references

## 🌍 Multi-language Support

### **Localization**
- Support for Turkish and English initially
- Use i18n libraries for text
- Consider right-to-left languages for future
- Maintain translation consistency

### **Research Papers**
- Accept contributions in Turkish and English
- Provide abstracts in both languages
- Follow academic formatting standards
- Include methodology in detail

## 🏆 Recognition

### **Contributors**
- All contributors listed in CONTRIBUTORS.md
- Research contributors credited in papers
- Major contributors invited to co-authorship
- Community recognition for significant contributions

### **Citation**
If you use this research in academic work:
```bibtex
@software{turkish_stock_market_ai,
  title={Turkish Stock Market AI Research (MAMUT R600)},
  author={[Contributors]},
  year={2024},
  url={https://github.com/[username]/Turkish-Stock-Market-AI-Research}
}
```

## 🔒 Security

### **Vulnerability Reporting**
- Report security issues privately
- Use GitHub Security Advisories
- Allow reasonable time for fixes
- Credit security researchers appropriately

### **Data Privacy**
- Follow GDPR and KVKK regulations
- Implement privacy by design
- Use differential privacy in ML models
- Document data handling procedures

## 📞 Communication

### **Channels**
- **Issues**: Technical discussions and bug reports
- **Discussions**: Research questions and ideas
- **Pull Requests**: Code reviews and improvements
- **Email**: Security issues and private matters

### **Code of Conduct**
- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and learners
- Maintain professional communication
- Respect diverse perspectives and backgrounds

## 🎯 Roadmap Contributions

### **Current Priorities**
1. Multi-language sentiment analysis expansion
2. TradingView REST API integration
3. Performance optimization improvements
4. Mobile-responsive dashboard enhancements

### **Future Directions**
1. Global market data integration
2. Advanced ML model architectures
3. Real-time trading strategy development
4. Regulatory compliance automation

---

Thank you for contributing to advancing Turkish financial market AI research! Your contributions help make financial technology more accessible and powerful for the Turkish market and beyond.

For questions, please open an issue or start a discussion in the repository.
