local wc = import "wirecell.jsonnet";
local g = import 'pgraph.jsonnet';

function(depofile) {
    local depos = g.pnode({
        type: 'NumpyDepoLoader',
        data: {
            filename: depofile
        }
    }, nin=0, nout=1),

    local random = {
        type: "Random",
        data: {
            generator: "default",
            seeds: [0,1,2,3,4],
        }
    },

    local lar = {
        // Longitudinal diffusion constant
        DL :  7.2 * wc.cm2/wc.s,
        // Transverse diffusion constant
        DT : 12.0 * wc.cm2/wc.s,
        // Electron lifetime
        lifetime : 8*wc.ms,
        // Electron drift speed, assumes a certain applied E-field
        drift_speed : 1.6*wc.mm/wc.us, // at 500 V/cm
        // LAr density
        density: 1.389*wc.g/wc.centimeter3,
        // Decay rate per mass for natural Ar39.
        ar39activity: 1*wc.Bq/wc.kg,
    },

    local drifter = g.pnode({
        local xregions = wc.unique_list([
            {anode: 0, response: 10*wc.cm, cathode: 1652},
            null
        ]),

        type: "Drifter",
        data: lar {
            rng: wc.tn(random),
            xregions: xregions,
            time_offset: 0.0,

            fluctuate: true, 
        },
    }, nin=1, nout=1, uses=[random]),    

    local dump = g.pnode({ type: "DumpDepos" }, nin=1, nout=0),

    local graph = g.pipeline([depos, drifter, dump]),

    local app = {
        type: 'Pgrapher',
        data: {
            edges: g.edges(graph)
        },
    },
    local cmdline = {
        type: "wire-cell",
        data: {
            plugins: ["WireCellSio", "WireCellGen",
                      "WireCellApps", "WireCellPgraph"],
            apps: ["Pgrapher"],
        }
    },
    seq: [cmdline] + g.uses(graph) + [app],
}.seq

