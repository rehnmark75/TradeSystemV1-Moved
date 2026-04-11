"use strict";(()=>{var e={};e.id=4582,e.ids=[4582],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},7377:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>k,patchFetch:()=>R,requestAsyncStorage:()=>d,routeModule:()=>m,serverHooks:()=>g,staticGenerationAsyncStorage:()=>E});var s={};r.r(s),r.d(s,{GET:()=>u,dynamic:()=>_});var c=r(7599),o=r(4294),n=r(4588),a=r(2921),i=r(740);let _="force-dynamic",l={Technology:"XLK","Health Care":"XLV",Financials:"XLF","Consumer Discretionary":"XLY","Communication Services":"XLC",Industrials:"XLI","Consumer Staples":"XLP",Energy:"XLE",Utilities:"XLU","Real Estate":"XLRE",Materials:"XLB"},p={Technology:["Technology"],"Health Care":["Health Care","Healthcare"],Financials:["Financials","Financial Services"],"Consumer Discretionary":["Consumer Discretionary","Consumer Cyclical"],"Consumer Staples":["Consumer Staples","Consumer Defensive"],"Communication Services":["Communication Services"],Industrials:["Industrials"],Energy:["Energy"],Utilities:["Utilities"],"Real Estate":["Real Estate"],Materials:["Materials","Basic Materials"]};async function u(){let e=await i.d.connect();try{let t=`
      SELECT
        sector,
        sector_etf,
        sector_return_1d,
        sector_return_5d,
        sector_return_20d,
        rs_vs_spy,
        rs_percentile,
        rs_trend,
        stocks_in_sector,
        pct_above_sma50,
        pct_bullish_trend,
        top_stocks,
        sector_stage
      FROM sector_analysis
      WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
      ORDER BY rs_vs_spy DESC
    `,r=await e.query(t);if(r.rows.length){let t=r.rows.map(e=>{if(e.top_stocks&&"string"==typeof e.top_stocks)try{e.top_stocks=JSON.parse(e.top_stocks)}catch(t){e.top_stocks=[]}return Array.isArray(e.top_stocks)||(e.top_stocks=[]),e}),s=t.filter(e=>!e.top_stocks||0===e.top_stocks.length).map(e=>e.sector).filter(Boolean);if(s.length){let r={},c=s.flatMap(e=>{let t=p[e]||[e];return t.forEach(t=>{r[t]=e}),t}),o=`
          SELECT
            i.sector,
            m.ticker,
            m.rs_percentile,
            m.rs_trend,
            m.current_price,
            ROW_NUMBER() OVER (PARTITION BY i.sector ORDER BY m.rs_percentile DESC) as rn
          FROM stock_screening_metrics m
          JOIN stock_instruments i ON m.ticker = i.ticker
          WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            AND m.rs_percentile IS NOT NULL
            AND i.sector = ANY($1::text[])
        `,n=await e.query(o,[c]),a={};n.rows.forEach(e=>{if(e.rn>5)return;let t=r[e.sector]||e.sector;a[t]||(a[t]=[]),a[t].push({ticker:e.ticker,rs_percentile:e.rs_percentile,rs_trend:e.rs_trend,price:e.current_price})}),t.forEach(e=>{e.top_stocks&&0!==e.top_stocks.length||(e.top_stocks=a[e.sector]||[])})}return a.NextResponse.json({rows:t})}let s=`
      SELECT
        i.sector,
        COUNT(*) as stocks_in_sector,
        AVG(m.rs_vs_spy) as avg_rs,
        AVG(m.rs_percentile) as avg_rs_percentile,
        COUNT(*) FILTER (WHERE m.current_price > m.sma_50) * 100.0 / NULLIF(COUNT(*), 0) as pct_above_sma50,
        COUNT(*) FILTER (WHERE m.trend_strength IN ('strong_up', 'up')) * 100.0 / NULLIF(COUNT(*), 0) as pct_bullish_trend,
        AVG(m.price_change_1d) as sector_return_1d,
        AVG(m.price_change_5d) as sector_return_5d,
        AVG(m.price_change_20d) as sector_return_20d
      FROM stock_instruments i
      JOIN stock_screening_metrics m ON i.ticker = m.ticker
      WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        AND i.sector IS NOT NULL
        AND i.sector <> ''
      GROUP BY i.sector
      ORDER BY avg_rs DESC NULLS LAST
    `,c=await e.query(s);if(!c.rows.length)return a.NextResponse.json({rows:[]});let o=`
      SELECT
        i.sector,
        m.ticker,
        m.rs_percentile,
        m.rs_trend,
        m.current_price,
        ROW_NUMBER() OVER (PARTITION BY i.sector ORDER BY m.rs_percentile DESC) as rn
      FROM stock_screening_metrics m
      JOIN stock_instruments i ON m.ticker = i.ticker
      WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        AND m.rs_percentile IS NOT NULL
    `,n=await e.query(o),i={};n.rows.forEach(e=>{e.rn>5||(i[e.sector]||(i[e.sector]=[]),i[e.sector].push({ticker:e.ticker,rs_percentile:e.rs_percentile,rs_trend:e.rs_trend,price:e.current_price}))});let _=c.rows.map(e=>{let t=e.avg_rs,r=e.pct_bullish_trend||0,s="lagging";return s=t&&t>1?r>50?"leading":"weakening":r>40?"improving":"lagging",{sector:e.sector,sector_etf:l[e.sector]||"",rs_vs_spy:e.avg_rs,rs_percentile:e.avg_rs_percentile?Math.round(e.avg_rs_percentile):null,rs_trend:"stable",sector_return_1d:e.sector_return_1d,sector_return_5d:e.sector_return_5d,sector_return_20d:e.sector_return_20d,stocks_in_sector:e.stocks_in_sector,pct_above_sma50:e.pct_above_sma50,pct_bullish_trend:e.pct_bullish_trend,sector_stage:s,top_stocks:i[e.sector]||[]}});return a.NextResponse.json({rows:_})}catch(e){return a.NextResponse.json({error:"Failed to load sector analysis"},{status:500})}finally{e.release()}}let m=new c.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/market/sectors/route",pathname:"/api/market/sectors",filename:"route",bundlePath:"app/api/market/sectors/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/market/sectors/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:d,staticGenerationAsyncStorage:E,serverHooks:g}=m,k="/api/market/sectors/route";function R(){return(0,n.patchFetch)({serverHooks:g,staticGenerationAsyncStorage:E})}},740:(e,t,r)=>{r.d(t,{d:()=>c}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let s=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",c=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:s,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),s=t.X(0,[5822,9967],()=>r(7377));module.exports=s})();