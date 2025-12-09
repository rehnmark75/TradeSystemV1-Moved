"""
Stock Prompt Builder

Builds institutional-grade prompts for Claude API stock signal analysis.
Combines technical, fundamental, and risk data into structured prompts.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StockPromptBuilder:
    """
    Builds prompts for Claude API stock analysis.

    Supports different analysis levels:
    - quick: Fast assessment, minimal context (~500 tokens)
    - standard: Balanced analysis (~800 tokens)
    - comprehensive: Full institutional analysis (~1200 tokens)
    """

    def __init__(self):
        self.analysis_levels = ['quick', 'standard', 'comprehensive']

    def build_signal_analysis_prompt(
        self,
        signal: Dict[str, Any],
        technical_data: Dict[str, Any],
        fundamental_data: Dict[str, Any],
        analysis_level: str = 'standard'
    ) -> str:
        """
        Build a comprehensive analysis prompt for a stock signal.

        Args:
            signal: Signal data (entry, stop, targets, score, etc.)
            technical_data: Technical indicators and metrics
            fundamental_data: Fundamental metrics and company data
            analysis_level: 'quick', 'standard', or 'comprehensive'

        Returns:
            Formatted prompt string for Claude API
        """
        if analysis_level not in self.analysis_levels:
            analysis_level = 'standard'

        # Build sections based on analysis level
        if analysis_level == 'quick':
            return self._build_quick_prompt(signal, technical_data, fundamental_data)
        elif analysis_level == 'comprehensive':
            return self._build_comprehensive_prompt(signal, technical_data, fundamental_data)
        else:
            return self._build_standard_prompt(signal, technical_data, fundamental_data)

    def _build_quick_prompt(
        self,
        signal: Dict[str, Any],
        technical: Dict[str, Any],
        fundamental: Dict[str, Any]
    ) -> str:
        """Build a quick assessment prompt (~500 tokens)"""

        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('signal_type', 'BUY')
        scanner = signal.get('scanner_name', 'unknown')
        score = signal.get('composite_score', 0)
        tier = signal.get('quality_tier', 'C')

        # Key metrics
        entry = self._format_price(signal.get('entry_price'))
        stop = self._format_price(signal.get('stop_loss'))
        target = self._format_price(signal.get('take_profit_1'))
        rr_ratio = signal.get('risk_reward_ratio', 0)

        # Technical summary
        rsi = technical.get('rsi_14', 50)
        trend = technical.get('trend_strength', 'neutral')
        rel_vol = technical.get('relative_volume', 1.0)

        # Fundamental summary
        pe = fundamental.get('trailing_pe', 'N/A')
        earnings_days = fundamental.get('days_to_earnings', 'N/A')

        prompt = f"""Analyze this stock signal quickly.

SIGNAL: {ticker} {direction} | Scanner: {scanner} | Score: {score}/100 ({tier})
LEVELS: Entry ${entry} | Stop ${stop} | Target ${target} | R:R {rr_ratio:.1f}:1
TECHNICAL: RSI {rsi:.0f} | Trend: {trend} | Volume: {rel_vol:.1f}x
FUNDAMENTAL: P/E {pe} | Earnings in: {earnings_days} days

Respond in JSON only:
{{"grade":"A+/A/B/C/D","score":1-10,"action":"STRONG BUY/BUY/HOLD/AVOID","thesis":"One sentence","conviction":"HIGH/MEDIUM/LOW"}}"""

        return prompt

    def _build_standard_prompt(
        self,
        signal: Dict[str, Any],
        technical: Dict[str, Any],
        fundamental: Dict[str, Any]
    ) -> str:
        """Build a standard analysis prompt (~800 tokens)"""

        ticker = signal.get('ticker', 'UNKNOWN')
        company_name = fundamental.get('name', ticker)
        direction = signal.get('signal_type', 'BUY')
        scanner = signal.get('scanner_name', 'unknown')
        score = signal.get('composite_score', 0)
        tier = signal.get('quality_tier', 'C')

        # Risk parameters
        entry = self._format_price(signal.get('entry_price'))
        stop = self._format_price(signal.get('stop_loss'))
        target1 = self._format_price(signal.get('take_profit_1'))
        target2 = self._format_price(signal.get('take_profit_2'))
        risk_pct = signal.get('risk_percent', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)

        # Technical section
        technical_section = self._format_technical_section(technical)

        # Fundamental section
        fundamental_section = self._format_fundamental_section(fundamental)

        # Confluence factors
        factors = signal.get('confluence_factors', [])
        factors_str = ', '.join(factors[:6]) if factors else 'None listed'

        prompt = f"""You are a Senior Equity Analyst. Analyze this stock signal with institutional rigor.

## SIGNAL OVERVIEW
**{ticker}** ({company_name}) | {direction} Signal
Scanner: {scanner} | Quality: {tier} ({score}/100)

## RISK PARAMETERS
Entry: ${entry} | Stop: ${stop} | Target 1: ${target1} | Target 2: ${target2}
Risk: {risk_pct:.1f}% | Reward/Risk: {rr_ratio:.1f}:1

## TECHNICAL ANALYSIS
{technical_section}

## FUNDAMENTAL ANALYSIS
{fundamental_section}

## CONFLUENCE FACTORS
{factors_str}

---
Analyze and respond in this exact JSON format:
{{
  "grade": "A+/A/B/C/D",
  "score": 1-10,
  "conviction": "HIGH/MEDIUM/LOW",
  "action": "STRONG BUY/BUY/HOLD/AVOID",
  "key_strengths": ["strength1", "strength2"],
  "key_risks": ["risk1", "risk2"],
  "thesis": "2-3 sentence investment thesis explaining your reasoning",
  "position_recommendation": "Full/Half/Quarter/Skip",
  "stop_adjustment": "Tighten/Keep/Widen",
  "time_horizon": "Intraday/Swing/Position"
}}"""

        return prompt

    def _build_comprehensive_prompt(
        self,
        signal: Dict[str, Any],
        technical: Dict[str, Any],
        fundamental: Dict[str, Any]
    ) -> str:
        """Build a comprehensive institutional prompt (~1200 tokens)"""

        ticker = signal.get('ticker', 'UNKNOWN')
        company_name = fundamental.get('name', ticker)
        sector = fundamental.get('sector', 'Unknown')
        industry = fundamental.get('industry', 'Unknown')
        direction = signal.get('signal_type', 'BUY')
        scanner = signal.get('scanner_name', 'unknown')
        score = signal.get('composite_score', 0)
        tier = signal.get('quality_tier', 'C')
        setup_desc = signal.get('setup_description', '')
        market_regime = signal.get('market_regime', 'unknown')

        # Risk parameters
        entry = self._format_price(signal.get('entry_price'))
        stop = self._format_price(signal.get('stop_loss'))
        target1 = self._format_price(signal.get('take_profit_1'))
        target2 = self._format_price(signal.get('take_profit_2'))
        risk_pct = signal.get('risk_percent', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)

        # Score breakdown
        trend_score = signal.get('trend_score', 0)
        momentum_score = signal.get('momentum_score', 0)
        volume_score = signal.get('volume_score', 0)
        pattern_score = signal.get('pattern_score', 0)
        confluence_score = signal.get('confluence_score', 0)

        # Technical section
        technical_section = self._format_technical_section(technical, detailed=True)

        # Fundamental section
        fundamental_section = self._format_fundamental_section(fundamental, detailed=True)

        # Confluence factors
        factors = signal.get('confluence_factors', [])
        factors_str = '\n'.join([f"- {f}" for f in factors]) if factors else '- None listed'

        prompt = f"""You are a Senior Equity Analyst at a quantitative trading firm. Provide institutional-grade analysis of this stock signal.

## SIGNAL OVERVIEW
**{ticker}** - {company_name}
Sector: {sector} | Industry: {industry}
Signal: {direction} | Scanner: {scanner}
Quality Tier: {tier} | Composite Score: {score}/100
Market Regime: {market_regime}

## SCORE BREAKDOWN
- Trend: {trend_score:.0f}/25
- Momentum: {momentum_score:.0f}/20
- Volume: {volume_score:.0f}/15
- Pattern: {pattern_score:.0f}/15
- Confluence: {confluence_score:.0f}/10

## RISK MANAGEMENT
Entry Price: ${entry}
Stop Loss: ${stop} ({risk_pct:.1f}% risk)
Target 1: ${target1}
Target 2: ${target2}
Risk/Reward: {rr_ratio:.1f}:1

## TECHNICAL ANALYSIS
{technical_section}

## FUNDAMENTAL ANALYSIS
{fundamental_section}

## CONFLUENCE FACTORS
{factors_str}

## SETUP DESCRIPTION
{setup_desc if setup_desc else 'Standard scanner setup'}

---
Provide comprehensive analysis in this exact JSON format:
{{
  "grade": "A+/A/B/C/D",
  "score": 1-10,
  "conviction": "HIGH/MEDIUM/LOW",
  "action": "STRONG BUY/BUY/HOLD/AVOID",
  "key_strengths": ["strength1", "strength2", "strength3"],
  "key_risks": ["risk1", "risk2", "risk3"],
  "thesis": "2-3 sentence investment thesis with specific reasoning",
  "position_recommendation": "Full/Half/Quarter/Skip",
  "stop_adjustment": "Tighten/Keep/Widen",
  "time_horizon": "Intraday/Swing/Position",
  "catalyst_watch": "Any upcoming catalysts to monitor",
  "alternative_entry": "Better entry level if current is suboptimal"
}}"""

        return prompt

    def _format_technical_section(
        self,
        technical: Dict[str, Any],
        detailed: bool = False
    ) -> str:
        """Format technical data into readable section"""

        # Price and trend
        price = self._format_price(technical.get('close', technical.get('current_price')))
        change_1d = technical.get('price_change_1d', 0)
        change_5d = technical.get('price_change_5d', 0)
        trend = technical.get('trend_strength', 'neutral')
        ma_align = technical.get('ma_alignment', 'mixed')

        # Moving averages
        sma_20 = self._format_price(technical.get('sma_20'))
        sma_50 = self._format_price(technical.get('sma_50'))
        sma_200 = self._format_price(technical.get('sma_200'))
        vs_sma20 = technical.get('price_vs_sma20', 0)
        vs_sma50 = technical.get('price_vs_sma50', 0)

        # Momentum
        rsi = technical.get('rsi_14', 50)
        macd_hist = technical.get('macd_histogram', 0)
        rsi_signal = technical.get('rsi_signal', 'neutral')
        macd_signal = technical.get('macd_cross_signal', 'neutral')

        # Volume
        rel_vol = technical.get('relative_volume', 1.0)
        vol_percentile = technical.get('percentile_volume', 50)

        # Volatility
        atr_pct = technical.get('atr_percent', 0)

        # Patterns
        pattern = technical.get('candlestick_pattern', 'none')
        high_low = technical.get('high_low_signal', 'neutral')
        gap = technical.get('gap_signal', 'none')
        sma_cross = technical.get('sma_cross_signal', 'none')

        if detailed:
            section = f"""Price: ${price} | 1D: {change_1d:+.1f}% | 5D: {change_5d:+.1f}%
Trend: {trend} | MA Alignment: {ma_align}
SMA20: ${sma_20} ({vs_sma20:+.1f}%) | SMA50: ${sma_50} ({vs_sma50:+.1f}%) | SMA200: ${sma_200}
RSI: {rsi:.0f} ({rsi_signal}) | MACD: {macd_hist:+.3f} ({macd_signal})
Volume: {rel_vol:.1f}x avg ({vol_percentile:.0f}th percentile) | ATR: {atr_pct:.1f}%
Pattern: {pattern} | 52W: {high_low} | Gap: {gap} | MA Cross: {sma_cross}"""
        else:
            section = f"""Price: ${price} ({change_1d:+.1f}% 1D) | Trend: {trend} | MA: {ma_align}
RSI: {rsi:.0f} | MACD: {macd_signal} | Volume: {rel_vol:.1f}x | ATR: {atr_pct:.1f}%
Pattern: {pattern} | 52W: {high_low}"""

        # Add SMC section if available
        smc = technical.get('smc', {})
        if smc:
            smc_section = self._format_smc_section(smc)
            section += f"\n{smc_section}"

        return section

    def _format_smc_section(
        self,
        smc: Dict[str, Any]
    ) -> str:
        """Format Smart Money Concepts data into readable section"""

        smc_trend = smc.get('smc_trend', 'N/A')
        smc_bias = smc.get('smc_bias', 'N/A')
        last_bos = smc.get('last_bos_type', 'N/A')
        last_bos_date = smc.get('last_bos_date')
        last_bos_date_str = last_bos_date.strftime('%Y-%m-%d') if last_bos_date else 'N/A'
        last_bos_price = smc.get('last_bos_price')
        last_bos_price_str = f"${last_bos_price:.2f}" if last_bos_price else 'N/A'

        zone = smc.get('premium_discount_zone', 'N/A')
        zone_pos = smc.get('zone_position')
        zone_pos_str = f"{zone_pos:.1f}%" if zone_pos else 'N/A'

        swing_high = smc.get('swing_high')
        swing_low = smc.get('swing_low')
        swing_high_str = f"${swing_high:.2f}" if swing_high else 'N/A'
        swing_low_str = f"${swing_low:.2f}" if swing_low else 'N/A'

        ob_type = smc.get('nearest_ob_type', 'N/A')
        ob_price = smc.get('nearest_ob_price')
        ob_price_str = f"${ob_price:.2f}" if ob_price else 'N/A'
        ob_dist = smc.get('nearest_ob_distance')
        ob_dist_str = f"{ob_dist:.1f}%" if ob_dist else 'N/A'

        confluence = smc.get('smc_confluence_score')
        confluence_str = f"{confluence:.0f}/100" if confluence else 'N/A'

        section = f"""SMC: Trend {smc_trend} | Bias {smc_bias} | Zone: {zone} ({zone_pos_str})
Last BOS: {last_bos} on {last_bos_date_str} at {last_bos_price_str}
Swing Range: {swing_low_str} - {swing_high_str}
Nearest OB: {ob_type} at {ob_price_str} ({ob_dist_str} away) | SMC Score: {confluence_str}"""

        return section

    def _format_fundamental_section(
        self,
        fundamental: Dict[str, Any],
        detailed: bool = False
    ) -> str:
        """Format fundamental data into readable section"""

        # Valuation
        pe = fundamental.get('trailing_pe')
        pe_str = f"{pe:.1f}" if pe else 'N/A'
        forward_pe = fundamental.get('forward_pe')
        forward_pe_str = f"{forward_pe:.1f}" if forward_pe else 'N/A'
        peg = fundamental.get('peg_ratio')
        peg_str = f"{peg:.2f}" if peg else 'N/A'
        pb = fundamental.get('price_to_book')
        pb_str = f"{pb:.1f}" if pb else 'N/A'

        # Growth
        earnings_growth = fundamental.get('earnings_growth')
        eg_str = f"{earnings_growth*100:+.0f}%" if earnings_growth else 'N/A'
        revenue_growth = fundamental.get('revenue_growth')
        rg_str = f"{revenue_growth*100:+.0f}%" if revenue_growth else 'N/A'

        # Profitability
        roe = fundamental.get('return_on_equity')
        roe_str = f"{roe*100:.0f}%" if roe else 'N/A'
        margin = fundamental.get('profit_margin')
        margin_str = f"{margin*100:.0f}%" if margin else 'N/A'

        # Health
        debt_eq = fundamental.get('debt_to_equity')
        de_str = f"{debt_eq:.1f}" if debt_eq else 'N/A'
        current = fundamental.get('current_ratio')
        cr_str = f"{current:.1f}" if current else 'N/A'

        # Ownership
        inst_pct = fundamental.get('institutional_percent')
        inst_str = f"{inst_pct:.0f}%" if inst_pct else 'N/A'
        short_pct = fundamental.get('short_percent_float')
        short_str = f"{short_pct:.1f}%" if short_pct else 'N/A'

        # Analyst
        rating = fundamental.get('analyst_rating', 'N/A')
        target = fundamental.get('target_price')
        target_str = f"${target:.2f}" if target else 'N/A'

        # Events
        earnings_date = fundamental.get('earnings_date')
        if earnings_date:
            if hasattr(earnings_date, 'strftime'):
                ed_str = earnings_date.strftime('%Y-%m-%d')
            else:
                ed_str = str(earnings_date)
        else:
            ed_str = 'N/A'

        days_to_earnings = fundamental.get('days_to_earnings')
        dte_str = f"{days_to_earnings} days" if days_to_earnings else 'N/A'

        # Dividend
        div_yield = fundamental.get('dividend_yield')
        div_str = f"{div_yield:.1f}%" if div_yield else '0%'

        if detailed:
            section = f"""VALUATION: P/E {pe_str} | Fwd P/E {forward_pe_str} | PEG {peg_str} | P/B {pb_str}
GROWTH: Earnings {eg_str} | Revenue {rg_str}
PROFITABILITY: ROE {roe_str} | Margin {margin_str}
HEALTH: D/E {de_str} | Current Ratio {cr_str}
OWNERSHIP: Institutional {inst_str} | Short Interest {short_str}
ANALYST: Rating {rating} | Target {target_str}
EVENTS: Earnings {ed_str} ({dte_str}) | Dividend {div_str}"""
        else:
            section = f"""P/E: {pe_str} | PEG: {peg_str} | Earnings Growth: {eg_str}
ROE: {roe_str} | D/E: {de_str} | Institutional: {inst_str} | Short: {short_str}
Analyst: {rating} | Earnings: {dte_str}"""

        return section

    def _format_price(self, price: Any) -> str:
        """Format price value safely"""
        if price is None:
            return 'N/A'
        try:
            return f"{float(price):.2f}"
        except (ValueError, TypeError):
            return 'N/A'

    def build_batch_summary_prompt(
        self,
        signals: List[Dict[str, Any]]
    ) -> str:
        """Build a prompt for summarizing multiple signals"""

        signal_summaries = []
        for s in signals[:10]:  # Limit to 10 for token efficiency
            ticker = s.get('ticker', '???')
            direction = s.get('signal_type', 'BUY')
            score = s.get('composite_score', 0)
            tier = s.get('quality_tier', 'C')
            rr = s.get('risk_reward_ratio', 0)
            signal_summaries.append(f"{ticker} {direction} - {tier}({score}) R:R {rr:.1f}")

        signals_str = '\n'.join(signal_summaries)

        prompt = f"""Review these {len(signals)} stock signals and rank the top 3:

{signals_str}

Respond in JSON:
{{"top_picks":[{{"ticker":"XXX","rank":1,"reason":"brief reason"}}],"market_read":"overall market assessment"}}"""

        return prompt
