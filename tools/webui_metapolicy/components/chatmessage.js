import React, {Component} from 'react';


class MessageItem extends Component {


    render() {
        return  <div className={this.props.party}> 
            {this.props.message}
            <p className="party">{this.props.party}</p>
        <style jsx>{`
            border-radius: 5px;
            padding: 10px;
            margin: 10px 20px;
            max-width: 75%;

            .party {
                color: #3e3e3e;
                font-size: 10px;
                padding: 1px;
                margin: 1px;
            }
            .system { 
                position: relative;
                background: white;
                border-radius: .4em;
            }
            .system:after {
                content: '';
                position: absolute;
                left: 0;
                top: 50%;
                width: 0;
                height: 0;
                border: 20px solid transparent;
                border-right-color: white;
                border-left: 0;
                border-top: 0;
                margin-top: -10px;
                margin-left: -20px;
            }

            .user { 
                position: relative;
                background: #349eeb;
                border-radius: .4em;
                justify-self: end;
                text-align: right;
            }
            .user:after {
                content: '';
                position: absolute;
                right: 0;
                top: 50%;
                width: 0;
                height: 0;
                border: 20px solid transparent;
                border-left-color: #349eeb;
                border-right: 0;
                border-top: 0;
                margin-top: -10px;
                margin-right: -20px;
            }
        `}
        </style>
        </div>
    }
}


export default MessageItem;