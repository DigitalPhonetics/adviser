import React, {Component} from 'react';

// import '../App.css';

class TurnInfoItem extends Component {

    router(key, data) {
        if(key === 'user_acts') {
            return this.render_nlu(data);
        } else if(key === 'beliefstate') {
            return this.render_bst(data);
        } else if(key === 'sys_act') {
            return this.render_policy(data);
        } else if(key === 'sys_utterance') {
            return this.render_nlg(data);
        }
    }

    render_nlu(nlu_data) {
        // nlu data is array of strings
        return <div> {
                nlu_data.map(userAct => <div>
                    <div className='item'><div className="header">{userAct.type} <i>(score: {userAct.score})</i></div>
                    {
                        userAct.slot != null && <div className='slotvaluecontainer'>
                            <div className="slot">{userAct.slot}</div>
                            <div className="value">{userAct.value}</div>
                        </div>
                    }
                   </div>
                </div>
                )
            }
        </div>
    }

    render_array(key, arr) {
        if(key instanceof Array) {
            return <div className="slotvaluecontainer">
                <div className="slot">{key[0]}</div>
                <div className="value">{key[1]}</div>
            </div>
        }
    }

    render_list(key, list) {
        return <div className="slotvaluecontainer">
            <div className="slot">{key}</div> 
            {
                list.map(value =>
                 <div className="value">{value}</div>
            )}
        </div>
    }

    render_bst_entry(bst_entry) {
        return <div>
            {
                Object.entries(bst_entry).map((key, idx) => (
                    this.render_array(key, bst_entry[key])
                ))
            }
        </div>
    }

    render_bst(bst_data) {
        return <div> 
            <div className="item">
            {
                Object.entries(bst_data.beliefs).map((bst_entry, index) => <div>
                    <div className="header">{bst_entry[0]}</div>
                    {this.render_bst_entry(bst_entry[1])}
                </div>
                )
            }
            </div>
            <div className="item">
            <div className="header">System</div>
                {
                    Object.entries(bst_data.system).map((bst_entry, index) => <div>
                        <div className="slotvaluecontainer">
                            <div className="slot">{bst_entry[0]}</div>
                            <div className="value">{bst_entry[1]}</div>
                        </div>
                    </div>
                    )
                }
            </div>
    </div>
    }

    render_sysact_values(values) {
        return <div> {
            Object.entries(values).map((key, idx) => 
            (
                key !== undefined && this.render_list(key[0], key[1])
            ))}
        </div>
    }

    render_policy(sys_act) {
        return <div className='item'>
            <div className="header">{sys_act.type}</div>
            {
                this.render_sysact_values(sys_act.values)
            }
        </div>
    }

    render_nlg(sys_utterance) {
        return <div className='item'>
            <div className="header">Utterance</div>
            <div className="slot">{sys_utterance}</div>
        </div>
    }

    render() {
        return  <div> 
            <div className='module'>
                <b>{this.props.name}</b>
                {
                    Object.keys(this.props.diff).map((key, index) => (
                        //this.render_router(key, this.props.diff[key])
                        this.router(key, this.props.diff[key])
                    ))
                }
            </div>
            <div className='arrow'>
                &darr;
            </div>
        </div>
    }
}


export default TurnInfoItem;