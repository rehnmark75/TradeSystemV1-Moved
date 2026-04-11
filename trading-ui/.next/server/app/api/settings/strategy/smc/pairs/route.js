"use strict";(()=>{var e={};e.id=9125,e.ids=[9125],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},3052:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>f,patchFetch:()=>y,requestAsyncStorage:()=>g,routeModule:()=>_,serverHooks:()=>E,staticGenerationAsyncStorage:()=>m});var s={};t.r(s),t.d(s,{GET:()=>c,POST:()=>l,dynamic:()=>u});var a=t(7599),o=t(4294),i=t(4588),n=t(2921),p=t(4372);let u="force-dynamic";async function d(e){let r=e??p.A,t=await r.query(`
      SELECT id
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `);return t.rows[0]?.id??null}async function c(){try{let e=await d();if(!e)return n.NextResponse.json({error:"No active SMC config found"},{status:404});let r=await p.A.query(`
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1
        ORDER BY epic ASC
      `,[e]);return n.NextResponse.json({config_id:e,overrides:r.rows??[]})}catch(e){return console.error("Failed to load pair overrides",e),n.NextResponse.json({error:"Failed to load pair overrides"},{status:500})}}async function l(e){let r=await e.json().catch(()=>null);if(!r||"object"!=typeof r)return n.NextResponse.json({error:"Invalid request body"},{status:400});let{epic:t,overrides:s,updates:a,updated_by:o,change_reason:i}=r;if(!t)return n.NextResponse.json({error:"epic is required"},{status:400});if(!o||!i)return n.NextResponse.json({error:"updated_by and change_reason are required"},{status:400});let u=await d();if(!u)return n.NextResponse.json({error:"No active SMC config found"},{status:404});try{let e=await p.A.query(`
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'smc_simple_pair_overrides'
      `),r=new Set(e.rows.map(e=>e.column_name).filter(e=>!["id","config_id","created_at","updated_at","updated_by","change_reason","epic"].includes(e))),d={...a??{}};void 0===d.parameter_overrides&&s&&(d.parameter_overrides=s);let c=Object.keys(d).filter(e=>r.has(e)),l=["config_id","epic",...c,"updated_by","change_reason"],_=[u,t];c.forEach(e=>_.push(d[e])),_.push(o),_.push(i);let g=l.map((e,r)=>`$${r+1}`),m=c.map(e=>`${e} = EXCLUDED.${e}`).concat(["updated_by = EXCLUDED.updated_by","change_reason = EXCLUDED.change_reason"]).join(", "),E=await p.A.query(`
        INSERT INTO smc_simple_pair_overrides
          (${l.join(", ")})
        VALUES (${g.join(", ")})
        ON CONFLICT (config_id, epic)
        DO UPDATE SET
          ${m}
        RETURNING *
      `,_);return await p.A.query(`
        INSERT INTO smc_simple_config_audit
          (config_id, pair_override_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, $2, 'PAIR_UPSERT', $3, $4, $5, $6)
      `,[u,E.rows[0]?.id??null,o,i,null,JSON.stringify(d)]),n.NextResponse.json(E.rows[0])}catch(e){return console.error("Failed to upsert pair override",e),n.NextResponse.json({error:"Failed to upsert pair override"},{status:500})}}let _=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/settings/strategy/smc/pairs/route",pathname:"/api/settings/strategy/smc/pairs",filename:"route",bundlePath:"app/api/settings/strategy/smc/pairs/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/settings/strategy/smc/pairs/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:g,staticGenerationAsyncStorage:m,serverHooks:E}=_,f="/api/settings/strategy/smc/pairs/route";function y(){return(0,i.patchFetch)({serverHooks:E,staticGenerationAsyncStorage:m})}},4372:(e,r,t)=>{t.d(r,{A:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let s=process.env.STRATEGY_CONFIG_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/strategy_config",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:s,max:10})}};var r=require("../../../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[5822,9967],()=>t(3052));module.exports=s})();